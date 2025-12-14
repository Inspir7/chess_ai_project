import chess
import torch
import numpy as np
import random

from training.mcts import MCTS
from training.move_encoding import move_to_index, get_total_move_count
from data.generate_labeled_data import fen_to_tensor

LENGTH_PENALTY = 0.002  # много малко


# ============================================================
# ILLEGAL-MOVE MASKING (за други места – тук не се ползва)
# ============================================================

def mask_illegal(policy_logits, board):
    """
    Получава 1D тензор policy_logits и занулява нелегални ходове чрез -inf логити.
    Връща softmax върху само легалните ходове.
    """
    import torch
    total = get_total_move_count()

    mask = torch.full((total,), float('-inf'), device=policy_logits.device)
    for mv in board.legal_moves:
        idx = move_to_index(mv)
        if idx is not None and 0 <= idx < total:
            mask[idx] = 0.0

    masked = policy_logits + mask

    if torch.isinf(masked).all():
        masked = torch.zeros_like(masked)

    return torch.softmax(masked, dim=0)


# ============================================================
# Temperature sampling
# ============================================================

def sample_move_from_pi(pi_dict, temperature=1.0):
    """Взема ход от π (visit-count distribution)."""
    moves = list(pi_dict.keys())
    probs = np.array(list(pi_dict.values()), dtype=np.float32)

    if probs.sum() <= 0:
        probs = np.ones(len(moves), dtype=np.float32) / len(moves)
    else:
        probs = probs / probs.sum()

    if temperature <= 1e-6:
        return moves[int(np.argmax(probs))]

    scaled = probs ** (1.0 / temperature)
    scaled /= scaled.sum()
    return random.choices(moves, weights=scaled, k=1)[0]


# ============================================================
# Проста материална хевристика при отрязване
# ============================================================

def static_material_eval(board: chess.Board) -> float:
    """
    Връща оценка от гледна точка на БЕЛИТЕ в [-1, 1] на база материал.
    Позитивно => белите водят, негативно => черните водят.
    """
    piece_values = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.0,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
    }

    material = 0.0
    for pt, val in piece_values.items():
        material += val * (
                len(board.pieces(pt, chess.WHITE)) -
                len(board.pieces(pt, chess.BLACK))
        )

    # Нормализация – да не избухва твърде много
    # (около +/-20 материал → tanh(~5) ≈ 1)
    return float(np.tanh(material / 6.0))


# ============================================================
# FULL SELF-PLAY EPISODE
# ============================================================

def play_episode(
        model,
        device,
        frozen_model=None,  # CHANGED: Приема frozen_model (supervised model)
        simulations=200,
        base_temperature=1.0,
        verbose=False,
        max_steps=160,
):
    """
    Връща:
        examples = [(state_np, pi_np, value_float)]
        result_str = "1-0"/"0-1"/"1/2-1/2"
        ply_count  = брой полуходове (plies)
    """
    board = chess.Board()

    # CHANGED: Конфигурация за асиметрична игра (Asymmetric Self-Play)
    use_frozen = (frozen_model is not None)
    # Ако има frozen модел, избираме на случаен принцип дали той да играе с Белите или Черните
    frozen_player_color = random.choice([chess.WHITE, chess.BLACK]) if use_frozen else None

    # Основен MCTS
    mcts_main = MCTS(model, device, simulations=simulations)

    # Втори MCTS за frozen модела (ако има такъв)
    if use_frozen:
        # Frozen играе с по-малко симулации (напр. 60% от основния), за да е малко по-слаб
        # и да даде шанс на основния модел да намери пролуки.
        frozen_sims = max(10, int(simulations * 0.6))
        mcts_frozen = MCTS(frozen_model, device, simulations=frozen_sims)
    else:
        mcts_frozen = None

    total_moves = get_total_move_count()
    examples = []
    ply = 0

    # ============================================================
    # PLAY LOOP
    # ============================================================
    while not board.is_game_over() and ply < max_steps:
        if ply < 20:
            T = base_temperature * 1.0
        elif ply < 60:
            T = base_temperature * 0.7
        else:
            T = base_temperature * 0.4

        # CHANGED: Избор на кой MCTS да се ползва
        is_frozen_turn = use_frozen and (board.turn == frozen_player_color)

        if is_frozen_turn:
            # Frozen моделът мисли
            pi_dict = mcts_frozen.run(board, move_number=ply)
        else:
            # Основният (learning) модел мисли
            pi_dict = mcts_main.run(board, move_number=ply)

        # π вектор (запазваме данни и от двамата, AlphaZero се учи и от опонента)
        pi_vector = np.zeros(total_moves, dtype=np.float32)
        for mv, p in pi_dict.items():
            idx = move_to_index(mv)
            if idx is not None and 0 <= idx < total_moves:
                pi_vector[idx] = p

        # Encode вход
        tens = fen_to_tensor(board.fen())
        arr = np.array(tens, dtype=np.float32)
        if arr.shape == (8, 8, 15):
            arr = arr.transpose(2, 0, 1)
        else:
            arr = arr.reshape(15, 8, 8)

        player = 1 if board.turn == chess.WHITE else -1
        examples.append((arr, pi_vector, player))

        # Избор на ход
        move = sample_move_from_pi(pi_dict, temperature=T)

        # Промоция винаги в царица
        if move.promotion is not None:
            move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)

        board.push(move)
        ply += 1

    # ============================================================
    # Финален резултат (CHANGED: Material-based scoring for draws)
    # ============================================================
    if board.is_game_over():
        r = board.result()
        if r == "1-0":
            final_z = 1.0
        elif r == "0-1":
            final_z = -1.0
        else:
            # CHANGED: Вместо 0.0 или -0.15, използваме материална оценка
            # Това дава градиент дори при реми! (Contempt for draw)
            mat_score = static_material_eval(board)  # [-1, 1]
            final_z = mat_score * 0.5  # Намалена тежест, защото все пак е реми
    else:
        # Truncated game → ползваме материала като оценка
        eval_z = static_material_eval(board)  # [-1, 1]
        final_z = eval_z  # Тук вярваме на материала напълно
        r = "*"

    # ============================================================
    # Присвояване на стойност на всяко състояние
    # ============================================================
    final_examples = []
    for (state_np, pi_np, player) in examples:
        # final_z е от гл.т. на белите (1.0 = White Win)
        # player е 1 за белите, -1 за черните
        # Ако final_z=1 (White Win) и player=1 (White turn) -> target=1
        # Ако final_z=1 (White Win) и player=-1 (Black turn) -> target=-1
        v = float(np.clip(final_z * player, -1.0, 1.0))
        final_examples.append((state_np, pi_np, v))

    if verbose:
        print(f"[Self-Play] result={r}, len={ply}, examples={len(final_examples)}, final_z={final_z:.3f}")

    return final_examples, r, ply