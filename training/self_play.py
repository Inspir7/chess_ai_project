import chess
import torch
import numpy as np
import random

from torch import optim  # вече не се използва, но го оставям за съвместимост
from models.AlphaZero import AlphaZeroModel
from training.mcts import MCTS
from training.move_encoding import move_to_index, get_total_move_count
from data.generate_labeled_data import fen_to_tensor
from training.buffer import ReplayBuffer


def sample_move_from_pi(pi_dict, temperature=1.0):
    """Семплира ход от pi_dict със зададена температура.
    Очакваме pi_dict да съдържа вероятности или visit-counts (нормализирани/ненормализирани).
    """
    # Поддържаме както probability-distribution, така и raw counts (те също работят)
    moves, probs = zip(*pi_dict.items())
    probs = np.array(probs, dtype=np.float32)

    # Ако вероятностите изглеждат като visit counts (не сумаризират до 1), нека ги нормализираме
    s = probs.sum()
    if s > 0:
        probs = probs / s
    else:
        # fallback: uniform
        probs = np.ones_like(probs, dtype=np.float32) / len(probs)

    if temperature == 0:
        return moves[np.argmax(probs)]

    scaled = np.power(probs, 1.0 / temperature)
    scaled_sum = scaled.sum()
    if scaled_sum > 0:
        scaled /= scaled_sum
    else:
        scaled = np.ones_like(scaled) / len(scaled)

    return random.choices(moves, weights=scaled, k=1)[0]


def play_episode(model, buffer, device, simulations=50, temperature=1.0, verbose=True):
    """
    Изиграва една self-play игра и събира (state, policy, value) примери в buffer.
    Внимание: buffer.push трябва да приема (state_tensor, pi_vector, value) или единичен triple.
    Алгоритъм (AlphaZero style):
      - за всяко състояние: записваме state и π (от visit counts / mcts distribution)
      - в края: определяме резултата z в {+1,-1,0} (от гледната точка на белите)
      - за всяка записана позиция: value = z * (1 if player_to_move == white else -1)
    """
    board = chess.Board()
    mcts = MCTS(model, device, simulations=simulations)
    game_history = []     # съхраняваме (state_tensor, pi_vector, player_to_move_sign)
    step = 0
    MAX_STEPS = 400

    while not board.is_game_over() and step < MAX_STEPS:
        pi_dict = mcts.run(board)

        # --- ensure pi_dict is usable: convert counts->probs if necessary (keep original items) ---
        # If mcts.run returned visit-counts, sample_move_from_pi and later normalization will handle it.
        # Convert pi_dict keys (moves) to vector
        pi_vector = torch.zeros(get_total_move_count(), dtype=torch.float32)
        total_prob = 0.0
        for move, prob in pi_dict.items():
            idx = move_to_index(move)
            if idx is None or idx < 0:
                continue
            try:
                pval = float(prob)
            except Exception:
                pval = 0.0
            pi_vector[idx] = pval
            total_prob += pval

        # Normalize pi_vector (PATCH) — гаранция, че пазим валидна probability distribution
        if total_prob > 0.0:
            pi_vector = pi_vector / float(total_prob)
        else:
            # fallback uniform over legal moves
            legal = list(board.legal_moves)
            if len(legal) > 0:
                uniform = 1.0 / len(legal)
                for mv in legal:
                    idx = move_to_index(mv)
                    if idx is not None and idx >= 0 and idx < len(pi_vector):
                        pi_vector[idx] = uniform

        # Encode current state
        tensor = fen_to_tensor(board.fen())
        if tensor.shape != (15, 8, 8):
            tensor = np.transpose(tensor, (2, 0, 1))
        state_tensor = torch.tensor(tensor, dtype=torch.float32)

        # Player perspective: +1 for white to move, -1 for black to move
        player = 1 if board.turn == chess.WHITE else -1

        # Append to local game history; do NOT yet assign final value
        game_history.append((state_tensor, pi_vector, player))

        # Select move according to pi_dict + temperature
        chosen_move = sample_move_from_pi(pi_dict, temperature=temperature)

        # Handle promotion sampling if necessary (keep existing logic)
        if chosen_move.promotion is not None:
            legal_promos = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
            promo_probs = np.array([
                pi_dict.get(chess.Move(chosen_move.from_square, chosen_move.to_square, promotion=p), 0.0)
                for p in legal_promos
            ], dtype=np.float32)
            s = promo_probs.sum()
            if s > 0:
                promo_probs = promo_probs / s
                chosen_move.promotion = np.random.choice(legal_promos, p=promo_probs)
            else:
                chosen_move.promotion = chess.QUEEN  # fallback

        # Push move and continue
        board.push(chosen_move)
        step += 1

    # Compute final reward z from White's perspective
    result_str = board.result()
    if result_str == "1-0":
        z = 1.0
    elif result_str == "0-1":
        z = -1.0
    else:
        z = 0.0

    # Assign value to all positions (AlphaZero perspective correction)
    # value_for_position = z * (1 if player_to_move==white else -1)
    for (s_t, pi_v, player) in game_history:
        value = z * player
        buffer.push(s_t, pi_v, value)

    if verbose:
        print(f"[Self-Play] Game finished: {result_str} → z: {z}, steps: {step}, positions: {len(game_history)}")
        try:
            with open("training_log.txt", "a") as f:
                f.write(f"[Self-Play] {result_str} | z={z} | steps={step} | positions={len(game_history)}\n")
        except Exception:
            pass

    return result_str


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroModel().to(device)
    buffer = ReplayBuffer()

    NUM_GAMES = 10
    for i in range(NUM_GAMES):
        print(f"=== Starting Self-Play Game {i+1}/{NUM_GAMES} ===")
        play_episode(model, buffer, device, simulations=100, temperature=1.0, verbose=True)