import chess
import torch
import numpy as np
import random
from torch import optim

from models.AlphaZero import AlphaZeroModel
from training.mcts import MCTS
from training.move_encoding import move_to_index, get_total_move_count
from training.generate_labeled_data import fen_to_tensor

def sample_move_from_pi(pi_dict, temperature=1.0):
    """Семплира ход от pi_dict със зададена температура (1.0 = нормално, <1 = по-решителен, >1 = по-хаотичен)."""
    moves, probs = zip(*pi_dict.items())
    probs = np.array(probs, dtype=np.float32)

    if temperature == 0:
        # Избор на най-добър ход (без стохастичност)
        return moves[np.argmax(probs)]

    # Прилагане на температура
    scaled = np.power(probs, 1.0 / temperature)
    scaled /= np.sum(scaled)

    return random.choices(moves, weights=scaled, k=1)[0]

def train_model(model, examples, optimizer, device):
    model.train()

    for state, policy, value in examples:
        state = state.unsqueeze(0).to(device)
        policy = policy.to(device)

        # to tensor
        value = torch.tensor(value, dtype=torch.float32).to(device)

        # prediction
        logits, predicted_value = model(state)

        # legal move s
        legal_move_indices = torch.nonzero(policy > 0, as_tuple=True)[0]  # Индекси на легалните ходове
        logits_legal = logits[0, legal_move_indices]  # Логиците само за легалните ходове
        policy_legal = policy[legal_move_indices]  # Политиката само за легалните ходове

        # funkciq na zagubata
        policy_loss = -torch.sum(policy_legal * torch.log_softmax(logits_legal, dim=0))  # Cross entropy loss
        value_loss = (predicted_value - value) ** 2  # MSE loss за стойността


        total_loss = policy_loss + value_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # log
        print(f"Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}")



def play_episode(model, device, simulations=50, temperature=1.0, verbose=True):
    board = chess.Board()
    mcts = MCTS(model, device, simulations=simulations)

    examples = []
    step = 0
    MAX_STEPS = 200  # безопасна граница

    if verbose:
        print("\n[Self-Play] New game started.")

    while not board.is_game_over() and step < MAX_STEPS:
        if verbose:
            print(f"\n[Step {step}] Current position:")
            print(board)

        pi_dict = mcts.run(board)

        # mcts policy
        sorted_pi = sorted(pi_dict.items(), key=lambda x: -x[1])
        print(f"[Step {step}] MCTS Policy: {[(str(mv), round(prob, 3)) for mv, prob in sorted_pi]}")

        legal_moves = list(pi_dict.keys())

        # policy -> vektor
        pi = np.zeros(get_total_move_count(), dtype=np.float32)
        for mv in legal_moves:
            idx = move_to_index(mv)
            if idx >= 0:
                pi[idx] = pi_dict[mv]

        chosen_move = sample_move_from_pi(pi_dict, temperature=temperature)
        print(f"[Step {step}] Chosen move: {chosen_move} (temp={temperature})")

        _, v = mcts._model_eval(board)
        print(f"[Step {step}] Predicted value: {v:.3f}")

        tensor = fen_to_tensor(board.fen())
        if tensor.shape != (15, 8, 8):
            tensor = np.transpose(tensor, (2, 0, 1))

        state_tensor = torch.tensor(tensor, dtype=torch.float32)
        examples.append((state_tensor, torch.tensor(pi), v))

        board.push(chosen_move)
        step += 1

        # pat, povtorenie
        if board.can_claim_threefold_repetition():
            print("[!] Threefold repetition is claimable.")
            break

        if board.can_claim_fifty_moves():
            print("[!] 50-move rule is claimable.")
            break

        if board.is_stalemate():
            print("[!] Stalemate detected.")
            break

        if board.is_insufficient_material():
            print("[!] Draw due to insufficient material.")
            break

    # reward
    result = board.result()
    reward = 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0.0

    print(f"\n[Self-Play] Game finished: {result} → reward: {reward}")

    final_examples = [(s, p, reward) for (s, p, _) in examples]

    # След всяка игра, извършваме тренировка
    return final_examples

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroModel().to(device)  # Зареди си модела тук
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.eval()

    # game i train
    for _ in range(10):  # 10 igri
        final_examples = play_episode(model, device, simulations=50, temperature=1.0, verbose=True)
        train_model(model, final_examples, optimizer, device)
        print("[Self-Play] Model updated after game.")
