# load trained model in order to self-play, creating new state, policy, value, which could be used for reLearning

import chess
import torch
import numpy as np
import random
import time
from torch import optim

from models.AlphaZero import AlphaZeroModel
from training.mcts import MCTS
from training.move_encoding import move_to_index, get_total_move_count
from data.generate_labeled_data import fen_to_tensor
from training.buffer import ReplayBuffer


def sample_move_from_pi(pi_dict, temperature=1.0):
    """Семплира ход от pi_dict със зададена температура."""
    if not pi_dict:
        return None
    moves, probs = zip(*pi_dict.items())
    probs = np.array(probs, dtype=np.float32)

    if temperature == 0 or np.sum(probs) == 0:
        return moves[np.argmax(probs)]

    scaled = np.power(probs, 1.0 / temperature)
    scaled /= np.sum(scaled)
    return random.choices(moves, weights=scaled, k=1)[0]


def play_episode(model, buffer, device, simulations=50, temperature=1.0, verbose=True):
    """Изиграва една self-play игра и тренира модела."""
    board = chess.Board()
    mcts = MCTS(model, device, simulations=simulations)
    step = 0
    MAX_STEPS = 200

    while not board.is_game_over() and step < MAX_STEPS:
        try:
            pi_dict = mcts.run(board)
        except Exception as e:
            print(f"[Self-Play] MCTS error: {e}")
            break

        if not pi_dict:
            # fallback при проблем или timeout
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            chosen_move = random.choice(legal_moves)
        else:
            chosen_move = sample_move_from_pi(pi_dict, temperature=temperature)
            if chosen_move is None:
                chosen_move = random.choice(list(board.legal_moves))

        # AI пешка промоция според policy
        if chosen_move.promotion is not None:
            legal_promos = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
            promo_probs = np.array([
                pi_dict.get(chess.Move(chosen_move.from_square, chosen_move.to_square, promotion=p), 0.0)
                for p in legal_promos
            ])
            if promo_probs.sum() > 0:
                promo_probs /= promo_probs.sum()
                chosen_move.promotion = np.random.choice(legal_promos, p=promo_probs)
            else:
                chosen_move.promotion = chess.QUEEN  # fallback

        tensor = fen_to_tensor(board.fen())
        if tensor.shape != (15, 8, 8):
            tensor = np.transpose(tensor, (2, 0, 1))
        state_tensor = torch.tensor(tensor, dtype=torch.float32)

        # Преобразуваме policy dict към вектор
        pi_vector = torch.zeros(get_total_move_count(), dtype=torch.float32)
        if pi_dict:
            for move, prob in pi_dict.items():
                idx = move_to_index(move)
                if idx >= 0:
                    pi_vector[idx] = prob

        turn_sign = 1.0 if board.turn else -1.0
        buffer.push(state_tensor, pi_vector, turn_sign)
        board.push(chosen_move)
        step += 1

    # Определяне на reward в края на играта
    result_str = board.result()
    reward = 1.0 if result_str == "1-0" else -1.0 if result_str == "0-1" else 0.0

    # Актуализиране на value във всички примери спрямо perspective:
    # buffer.buffer съдържа (state, pi, turn_sign) — където turn_sign е +1.0 или -1.0
    for i in range(len(buffer.buffer)):
        state, pi, turn_sign = buffer.buffer[i]
        try:
            # Ако случайно е None или не е числов, fallback към глобалния reward
            turn = float(turn_sign)
        except Exception:
            turn = 1.0
        final_val = reward * turn  # draw -> 0, win/lose -> +/-1 от perspective
        buffer.buffer[i] = (state, pi, final_val)

    # Обучение от буфера
    model.train()
    try:
        states, policies, values = buffer.sample(batch_size=32)
    except Exception:
        print("[Self-Play] Not enough samples for training yet.")
        return result_str

    loss_total = 0.0
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for state, pi_target, value_target in zip(states, policies, values):
        state = state.unsqueeze(0).to(device)
        pi_target_tensor = pi_target.to(device)

        # 🔧 поправка на предупреждението
        value_target_tensor = (
            value_target.detach().clone()
            if torch.is_tensor(value_target)
            else torch.tensor(value_target, dtype=torch.float32)
        ).unsqueeze(0).to(device)

        logits, predicted_value = model(state)
        policy_loss = -torch.sum(pi_target_tensor * torch.log_softmax(logits, dim=1))
        value_loss = (predicted_value - value_target_tensor) ** 2
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss.item()

    # Лог и запис
    if verbose and len(states) > 0:
        avg_loss = loss_total / len(states)
        print(f"[Self-Play] Game finished: {result_str} → reward: {reward}, Avg Loss: {avg_loss:.4f}")
        with open("training_log.txt", "a") as f:
            f.write(f"[Self-Play] {result_str} | reward={reward}, avg_loss={avg_loss:.4f}\n")

    # лека пауза за да избегнем закачане в multiproc
    time.sleep(0.2)
    return result_str


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroModel().to(device)
    buffer = ReplayBuffer()

    NUM_GAMES = 10
    for i in range(NUM_GAMES):
        print(f"=== Starting Self-Play Game {i+1}/{NUM_GAMES} ===")
        play_episode(model, buffer, device, simulations=50, temperature=1.0, verbose=True)