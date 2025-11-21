#тества върху етикетирани данни и извежда топ-1 / топ-3 accuracy и value loss.
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import re
from torch.nn import functional as F

from models.AlphaZero import AlphaZeroModel
from models.move_encoding import index_to_move


class ChessLabeledTensorDataset(torch.utils.data.Dataset):
    def __init__(self, states_array, policies_array, values_array):
        self.states = states_array
        self.policies = policies_array
        self.values = values_array

        assert len(self.states) == len(self.policies) == len(self.values), "Размерите не съвпадат."

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = torch.tensor(self.states[idx], dtype=torch.float32).permute(2, 0, 1)
        policy = torch.tensor(self.policies[idx], dtype=torch.float32)
        value = torch.tensor(self.values[idx], dtype=torch.float32)
        return state, policy, value


def load_model(model_path, device):
    model = AlphaZeroModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)


def load_multiple_npy(prefix: str, folder: Path):
    files = sorted(folder.glob(f"{prefix}*.npy"), key=lambda f: int(re.search(r'_(\d+)\.npy$', f.name).group(1)))
    arrays = [np.load(f) for f in files]
    return np.concatenate(arrays, axis=0)


def visualize_prediction(state_tensor, policy_pred, policy_target, value_pred, value_true):
    top_k = 3
    policy_pred = torch.softmax(policy_pred, dim=0).cpu().numpy()
    policy_target = policy_target.cpu().numpy()

    pred_top_k_indices = policy_pred.argsort()[-top_k:][::-1]
    target_index = np.argmax(policy_target)

    print("\n--- Prediction Example ---")
    print(f"True Value:     {value_true.item():.3f}")
    print(f"Predicted Value:{value_pred.item():.3f}")
    print(f"Target Move:    {index_to_move(target_index)}")
    print("Top 3 Predicted Moves:")
    for idx in pred_top_k_indices:
        print(f"  {index_to_move(idx)}  → prob={policy_pred[idx]:.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path("C:/Users/prezi/PycharmProjects/chess_ai_project/training/alpha_zero_supervised.pth")
    test_dir = Path("C:/Users/prezi/PycharmProjects/chess_ai_project/training/test")

    print("\n[Loading Test Data]")
    states = load_multiple_npy("test_labeled_states_", test_dir)
    policies = load_multiple_npy("test_labeled_policies_", test_dir)
    values = load_multiple_npy("test_labeled_values_", test_dir)

    print(f"Loaded shapes: states={states.shape}, policies={policies.shape}, values={values.shape}")

    dataset = ChessLabeledTensorDataset(states, policies, values)
    test_loader = DataLoader(dataset, batch_size=256, shuffle=False)

    print(f"\nLoaded {len(dataset)} test samples. Showing 5 random predictions:\n")

    model = load_model(model_path, device)

    total_value_loss = 0.0
    total_policy_loss = 0.0
    total_top1 = 0
    total_top3 = 0
    total_samples = 0

    with torch.no_grad():
        for state, policy_target, value_target in test_loader:
            state = state.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)

            policy_pred, value_pred = model(state)

            # MSE за value
            value_loss = F.mse_loss(value_pred.view(-1), value_target, reduction="sum").item()
            total_value_loss += value_loss

            # Cross entropy за policy
            policy_log_probs = torch.log_softmax(policy_pred, dim=1)
            policy_loss = -torch.sum(policy_log_probs * policy_target).item()
            total_policy_loss += policy_loss

            # Accuracy
            pred_topk = torch.topk(policy_pred, k=3, dim=1).indices
            true_move_idx = policy_target.argmax(dim=1)

            total_top1 += (pred_topk[:, 0] == true_move_idx).sum().item()
            total_top3 += ((pred_topk == true_move_idx.unsqueeze(1)).any(dim=1)).sum().item()

            total_samples += state.size(0)

    print("\n--- Evaluation Metrics ---")
    print(f"Value MSE Loss:     {total_value_loss / total_samples:.4f}")
    print(f"Policy CrossEntropy:{total_policy_loss / total_samples:.4f}")
    print(f"Top-1 Accuracy:     {total_top1 / total_samples:.4f}")
    print(f"Top-3 Accuracy:     {total_top3 / total_samples:.4f}")

    print("\nDone evaluating model.")




if __name__ == "__main__":
    main()
