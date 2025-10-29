# load trained model in order to self-play, creating new state, policy, value, which could be used for reLearning

import chess
import torch
import numpy as np
import random
from torch import optim

from models.AlphaZero import AlphaZeroModel
from training.mcts import MCTS
from training.move_encoding import move_to_index, get_total_move_count
from training.generate_labeled_data import fen_to_tensor
from training.buffer import ReplayBuffer


def sample_move_from_pi(pi_dict, temperature=1.0):
    """Семплира ход от pi_dict със зададена температура."""
    moves, probs = zip(*pi_dict.items())  # ✅ поправено
    probs = np.array(probs, dtype=np.float32)

    if temperature == 0:
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
        pi_dict = mcts.run(board)
        chosen_move = sample_move_from_pi(pi_dict, temperature=temperature)

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
        for move, prob in pi_dict.items():  # ✅ поправено
            idx = move_to_index(move)
            if idx >= 0:
                pi_vector[idx] = prob

        buffer.push(state_tensor, pi_vector, None)
        board.push(chosen_move)
        step += 1

    # Определяне на reward в края на играта
    result_str = board.result()
    reward = 1.0 if result_str == "1-0" else -1.0 if result_str == "0-1" else 0.0

    # Актуализиране на value във всички примери
    for i in range(len(buffer.buffer)):
        state, pi, _ = buffer.buffer[i]
        buffer.buffer[i] = (state, pi, reward)

    # Обучение от буфера
    model.train()
    states, policies, values = buffer.sample(batch_size=32)
    loss_total = 0.0
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for state, pi_target, value_target in zip(states, policies, values):
        state = state.unsqueeze(0).to(device)
        pi_target_tensor = pi_target.to(device)  # вече е тензор, няма нужда от dict
        value_target_tensor = torch.tensor(value_target, dtype=torch.float32).unsqueeze(0).to(device)

        logits, predicted_value = model(state)
        policy_loss = -torch.sum(pi_target_tensor * torch.log_softmax(logits, dim=1))
        value_loss = (predicted_value - value_target_tensor) ** 2
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss.item()

    # Лог и запис
    if verbose:
        avg_loss = loss_total / len(states)
        print(f"[Self-Play] Game finished: {result_str} → reward: {reward}, Avg Loss: {avg_loss:.4f}")
        with open("training_log.txt", "a") as f:
            f.write(f"[Self-Play] {result_str} | reward={reward}, avg_loss={avg_loss:.4f}\n")

    return result_str


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroModel().to(device)
    buffer = ReplayBuffer()

    NUM_GAMES = 10
    for i in range(NUM_GAMES):
        print(f"=== Starting Self-Play Game {i+1}/{NUM_GAMES} ===")
        play_episode(model, buffer, device, simulations=50, temperature=1.0, verbose=True)
