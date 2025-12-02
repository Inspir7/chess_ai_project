import chess
import torch
import numpy as np
import random

from training.mcts import MCTS
from training.move_encoding import move_to_index, get_total_move_count
from data.generate_labeled_data import fen_to_tensor


# ================================
# Utility: sample move from π
# ================================
def sample_move_from_pi(pi_dict, temperature=1.0):
    """
    Избира ход според π (visit-count policy), с температурно скалиране.
    """
    if not pi_dict:
        raise ValueError("sample_move_from_pi: empty policy dict")

    moves, probs = zip(*pi_dict.items())
    probs = np.array(probs, dtype=np.float32)

    s = probs.sum()
    if s > 0:
        probs /= s
    else:
        probs = np.ones_like(probs) / len(probs)

    if temperature <= 1e-6:
        return moves[int(np.argmax(probs))]

    scaled = np.power(probs, 1.0 / temperature)
    scaled_sum = scaled.sum()
    if scaled_sum <= 0:
        scaled = np.ones_like(probs) / len(probs)
    else:
        scaled /= scaled_sum

    return random.choices(moves, weights=scaled, k=1)[0]


# ================================
# Main self-play function
# ================================
def play_episode(
    model,
    buffer,
    device,
    simulations: int = 200,
    base_temperature: float = 1.0,
    verbose: bool = False,
    max_steps: int = 100,
    repetition_window: int = 12,
    draw_penalty: float = -0.3,
    reward_decay: float = 0.001,
    gamma: float = 0.99,
):
    """
    Изиграва една self-play партия с AlphaZero-подобен MCTS и записва (s, π, v) в буфера.
    """

    board = chess.Board()
    mcts = MCTS(model, device, simulations=simulations)

    total_moves = get_total_move_count()   # 4672 при новия encoding
    fen_history = []
    history = []
    ply = 0

    # =====================================================
    # SELF-PLAY LOOP
    # =====================================================
    while not board.is_game_over() and ply < max_steps:

        # -------------------------
        # Dynamic temperature
        # -------------------------
        if ply < 20:
            T = base_temperature * 1.8
        elif ply < 40:
            T = base_temperature * 0.5
        else:
            T = base_temperature * 0.15

        # ========================
        # MCTS → π (dict move→prob)
        # ========================
        pi_dict = mcts.run(board, move_number=ply)

        # Full π vector (size = 4672)
        pi_vector = torch.zeros(total_moves, dtype=torch.float32)

        for mv, p in pi_dict.items():
            idx = move_to_index(mv)
            if idx is not None and 0 <= idx < total_moves:
                pi_vector[idx] = float(p)

        # Normalize
        s = pi_vector.sum().item()
        if s > 0:
            pi_vector /= s
        else:
            # Uniform fallback over LEGAL moves
            legal = list(board.legal_moves)
            if len(legal) > 0:
                uni = 1.0 / len(legal)
                for mv in legal:
                    idx = move_to_index(mv)
                    if idx is not None and 0 <= idx < total_moves:
                        pi_vector[idx] = uni

        # ========================
        # Encode state → (15,8,8)
        # ========================
        tensor = fen_to_tensor(board.fen())
        tensor = np.array(tensor, copy=False)

        if tensor.shape == (8, 8, 15):
            tensor = np.transpose(tensor, (2, 0, 1))  # to CHW
        elif tensor.shape != (15, 8, 8):
            tensor = np.reshape(tensor, (15, 8, 8))

        state_tensor = torch.tensor(tensor, dtype=torch.float32)

        player = 1 if board.turn == chess.WHITE else -1

        history.append((state_tensor, pi_vector, player, ply))

        # ========================
        # Choose move
        # ========================
        move = sample_move_from_pi(pi_dict, temperature=T)

        # Promotion refinement
        if move.promotion is not None:
            promo_types = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
            promo_probs = []

            for pt in promo_types:
                alt = chess.Move(move.from_square, move.to_square, promotion=pt)
                promo_probs.append(pi_dict.get(alt, 0.0))

            promo_probs = np.array(promo_probs, dtype=np.float32)
            if promo_probs.sum() > 0:
                promo_probs /= promo_probs.sum()
                move.promotion = np.random.choice(promo_types, p=promo_probs)
            else:
                move.promotion = chess.QUEEN

        board.push(move)
        ply += 1

        # ========================
        # Anti-repetition
        # ========================
        fen_history.append(board.fen())
        if len(fen_history) > repetition_window:
            fen_history.pop(0)

        if fen_history.count(board.fen()) >= 3:
            if verbose:
                print("[Repetition → forced draw]")
            z = -0.25 * (1 if board.turn == chess.WHITE else -1)
            break

    # =====================================================
    # Final result → z
    # =====================================================
    if board.is_game_over():
        res = board.result()
        if res == "1-0":
            z = 1.0
        elif res == "0-1":
            z = -1.0
        else:
            z = draw_penalty
    else:
        z = draw_penalty

    # =====================================================
    # SHAPING & WRITE VALUES
    # =====================================================
    total_len = len(history)

    for s, pi, player, ply_idx in history:

        # Base signal
        if z == 1.0:
            base = 1.25
        elif z == -1.0:
            base = -1.25
        else:
            base = z

        v = base * player

        # Importance weighting for earlier moves
        early_boost = 1.0 + (0.8 * (1 - ply_idx / max(1, total_len)))
        v *= early_boost

        # Penalty for long games
        v *= max(0.0, 1 - reward_decay * ply_idx)

        # Discount (stabilises gradients)
        v *= gamma ** (ply_idx / max(1, 0.3 * total_len))

        v = float(np.clip(v, -1.5, 1.5))

        buffer.push(s, pi, v)

    if verbose:
        try:
            r = board.result()
        except:
            r = "N/A"
        print(f"[Self-Play] Result={r}, steps={ply}, positions={len(history)}")

    return board.result()