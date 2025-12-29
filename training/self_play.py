import chess
import torch
import numpy as np
import random

from training.mcts import MCTS
from training.move_encoding import move_to_index, get_total_move_count
from data.generate_labeled_data import fen_to_tensor


# ============================================================
# ILLEGAL-MOVE MASKING (запазваме го за съвместимост)
# ============================================================
def mask_illegal(policy_logits, board):
    """
    Получава 1D тензор policy_logits и занулява нелегални ходове.
    """
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
# HELPER: Material Evaluation
# ============================================================
def static_material_eval(board: chess.Board) -> float:
    """
    Връща оценка [-1, 1] от гледна точка на БЕЛИТЕ.
    Използва се за 'Adjudication' (служебна победа).
    """
    piece_values = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.1,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
    }

    # Броим материал
    w_mat = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in piece_values.items())
    b_mat = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in piece_values.items())

    diff = w_mat - b_mat

    # Нормализация:
    # 4 точки предимство (напр. лека фигура + пешка) дава ~0.76
    # 5 точки (Топ) дава ~0.85
    return float(np.tanh(diff / 4.0))


# ============================================================
# Temperature sampling
# ============================================================
def sample_move_from_pi(pi_dict, temperature=1.0):
    moves = list(pi_dict.keys())

    # --- FIX: Защита срещу празен MCTS ---
    if not moves:
        return None

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
# ADJUDICATION HELPER (Помощна функция за приключване)
# ============================================================
def _finish_with_adjudication(examples, winner_z, result_str, ply):
    """
    Приключва играта служебно и записва резултата за всички събрани стъпки.
    winner_z: 1.0 (Бели печелят) или -1.0 (Черни печелят)
    """
    final_examples = []
    for (state_np, pi_np, player) in examples:
        # player е 1 (Бели) или -1 (Черни)
        # Ако winner_z=1 (Бели) и player=1 (Бели) -> value = 1
        # Ако winner_z=1 (Бели) и player=-1 (Черни) -> value = -1
        v = float(np.clip(winner_z * player, -1.0, 1.0))
        final_examples.append((state_np, pi_np, v))

    return final_examples, result_str, ply


# ============================================================
# FULL SELF-PLAY EPISODE
# ============================================================
def play_episode(
        model,
        device,
        frozen_model=None,
        simulations=200,
        base_temperature=1.0,
        verbose=False,
        max_steps=220,  # Увеличено, за да има време за игра
):
    board = chess.Board()

    # 1. Настройка на моделите (Main vs Frozen)
    use_frozen = (frozen_model is not None)
    # Случайно избираме кой цвят да е "слабият" (Frozen) модел
    frozen_player_color = random.choice([chess.WHITE, chess.BLACK]) if use_frozen else None

    # Основен модел (Тигъра) - пълни симулации
    mcts_main = MCTS(model, device, simulations=simulations)

    # Frozen модел (Опонента) - намалени симулации (по-слаб)
    if use_frozen:
        frozen_sims = max(10, int(simulations * 0.5))  # 50% от силата
        mcts_frozen = MCTS(frozen_model, device, simulations=frozen_sims)
    else:
        mcts_frozen = None

    total_moves = get_total_move_count()
    examples = []
    ply = 0

    # Праг за служебна победа (Adjudication threshold)
    # 0.75 отговаря на ~4 пешки предимство.
    #MERCY_THRESHOLD = 0.75
    # НОВО: 0.98 (практически трябва да си взел всичко, за да спре играта без мат)
    # СТАРО:
    # MERCY_THRESHOLD = 0.98
    # (Това изисква да си взел ВСИЧКО, за да ти даде служебна победа)

    # НОВО:
    #MERCY_THRESHOLD = 0.85
    # (Това е равно на предимство от Топ или повече.
    # Ако AI поведе с Топ, спираме играта и му казваме "ТИ ПОБЕДИ".
    # Така то ще се научи, че материалното предимство води директно до победа.)

    # СТАРО: MERCY_THRESHOLD = 0.85

    # НОВО: Фаза 2 - Killer Instinct
    # Слагаме го на 0.99. Това означава, че играта спира само ако имаме Дама + Топове повече.
    # За всичко по-малко (примерно само един Топ повече) - ТРЯБВА ДА МАТИРА!
    MERCY_THRESHOLD = 0.99

    # ================= GAME LOOP =================
    while not board.is_game_over() and ply < max_steps:

        # --- A) Проверка за Adjudication (Служебна победа) ---
        # Правим го само след 20-тия ход, за да не прекъсваме гамбити в дебюта
        if ply > 20:
            mat_score = static_material_eval(board)  # Връща (white_advantage)

            # Ако Белите водят с много
            if mat_score > MERCY_THRESHOLD:
                return _finish_with_adjudication(examples, 1.0, "1-0 (Adj)", ply)

            # Ако Черните водят с много (mat_score е отрицателен)
            elif mat_score < -MERCY_THRESHOLD:
                return _finish_with_adjudication(examples, -1.0, "0-1 (Adj)", ply)

        # --- B) Температура (Exploration) ---
        if ply < 30:
            T = base_temperature
        else:
            # БЕШЕ: T = base_temperature * 0.5

            # НОВО: Твърд таван от 0.2 след 30-тия ход.
            # Това казва: "Стига глупости, намери най-добрия ход и го матирай!"
            T = min(base_temperature * 0.5, 0.2)

        # --- C) Избор кой MCTS да мисли ---
        is_frozen_turn = use_frozen and (board.turn == frozen_player_color)

        if is_frozen_turn:
            pi_dict = mcts_frozen.run(board, move_number=ply)
        else:
            pi_dict = mcts_main.run(board, move_number=ply)

        # --- D) Запис на данните ---
        pi_vector = np.zeros(total_moves, dtype=np.float32)
        for mv, p in pi_dict.items():
            idx = move_to_index(mv)
            if idx is not None and 0 <= idx < total_moves:
                pi_vector[idx] = p

        tens = fen_to_tensor(board.fen())
        arr = np.array(tens, dtype=np.float32)
        # Оправяне на дименсиите (Channel First)
        if arr.shape == (8, 8, 15):
            arr = arr.transpose(2, 0, 1)
        else:
            arr = arr.reshape(15, 8, 8)

        player = 1 if board.turn == chess.WHITE else -1
        examples.append((arr, pi_vector, player))

        # --- E) Изиграване на ход ---
        move = sample_move_from_pi(pi_dict, temperature=T)

        # =========================================================
        # ЗАЩИТНА СТЕНА (THE FIREWALL)
        # =========================================================

        # 1. Проверка дали ходът е легален на ТАЗИ дъска
        if move is not None:
            if move not in board.legal_moves:
                # Тук хващаме "призрачните" ходове (g2g4 като черни)
                # print(f"[WARN] Illegal move intercepted: {move}") # Може да разкоментираш за дебъг
                move = None  # Изхвърляме го и отиваме на fallback

        # 2. Fallback: Ако няма ход или е бил нелегален -> Случаен
        if move is None:
            if board.legal_moves.count() > 0:
                move = random.choice(list(board.legal_moves))
            else:
                break  # Мат или Пат

        # 3. Промоция (за всеки случай)
        if move.promotion is not None:
            move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)

        # 4. Изпълнение
        board.push(move)
        ply += 1

        # ================= END OF GAME =================
        if board.is_game_over():
            r = board.result()
            if r == "1-0":
                final_z = 1.0
            elif r == "0-1":
                final_z = -1.0
            else:
                '''При реми все пак гледаме материала
                #mat_score = static_material_eval(board)
                #final_z = mat_score * 0.5'''

                # === ТУК Е МАГИЯТА ЗА ФАЗА 2 ===
                # Реми (Stalemate, 3-fold repetition, 50-move rule)
                mat_score = static_material_eval(board)  # Връща позитивнo за белите, негативно за черните

                # Логика: "Ако имаш материално предимство, но направиш реми -> ГУБИШ ТОЧКИ"
                # 0.4 съответства на около 2 пешки предимство.

                if abs(mat_score) > 0.4:
                    # Наказваме страната, която има материал, но не е успяла да бие
                    # Вместо 0.0, даваме -0.5 (половин загуба)
                    final_z = -0.8 * np.sign(mat_score)
                else:
                    # Ако материалът е равен, ремито е честно (0.0)
                    final_z = 0.0

    else:
        # Timeout (Truncated) - не е стигнал threshold-а, но времето свърши
        # Оценяваме позицията по материал
        mat_score = static_material_eval(board)
        final_z = mat_score
        r = "*"

    # Присвояване на резултата към всички стъпки
    final_examples = []
    for (state_np, pi_np, player) in examples:
        v = float(np.clip(final_z * player, -1.0, 1.0))
        final_examples.append((state_np, pi_np, v))

    if verbose:
        print(f"[Self-Play] result={r}, len={ply}, final_z={final_z:.3f}")

    return final_examples, r, ply