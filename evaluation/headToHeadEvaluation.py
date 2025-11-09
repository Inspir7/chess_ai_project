import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torch.nn.functional as F
from models.AlphaZero import AlphaZeroModel


# --- Dataset class ---
class ChessLabeledTensorDataset(Dataset):
    def __init__(self, states, policies, values):
        self.states = states
        self.policies = policies
        self.values = values

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = torch.tensor(self.states[idx], dtype=torch.float32).permute(2, 0, 1)
        policy = torch.tensor(self.policies[idx], dtype=torch.float32)
        value = torch.tensor(self.values[idx], dtype=torch.float32)
        return state, policy, value


# --- Evaluation function ---
def evaluate_model(model, dataloader, device):
    total_value_loss = 0.0
    total_policy_loss = 0.0
    correct_top1, correct_top3, total = 0, 0, 0

    with torch.no_grad():
        for state, policy_target, value_target in dataloader:
            state = state.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)

            policy_pred, value_pred = model(state)

            value_loss = F.mse_loss(value_pred.squeeze(), value_target.squeeze(), reduction='sum')
            policy_log = torch.log_softmax(policy_pred, dim=1)
            policy_loss = -(policy_target * policy_log).sum()

            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()

            topk = torch.topk(policy_pred, k=3, dim=1).indices
            target_idx = policy_target.argmax(dim=1)
            correct_top1 += (topk[:, 0] == target_idx).sum().item()
            correct_top3 += ((topk == target_idx.unsqueeze(1)).any(dim=1)).sum().item()
            total += state.size(0)

    return {
        "value_loss": total_value_loss / total,
        "policy_loss": total_policy_loss / total,
        "top1_acc": correct_top1 / total * 100,
        "top3_acc": correct_top3 / total * 100,
    }


# --- Main ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    val_dir = Path("/home/presi/projects/chess_ai_project/training/validation")
    model_a_path = Path("/home/presi/projects/chess_ai_project/training/alpha_zero_supervised_ep5.pth")
    model_b_path = Path("/home/presi/projects/chess_ai_project/training/alpha_zero_supervised.pth")


    # Load validation data
    print("[INFO] Loading validation data...")
    states = np.load(val_dir / "val_labeled_states_0.npy")
    policies = np.load(val_dir / "val_labeled_policies_0.npy")
    values = np.load(val_dir / "val_labeled_values_0.npy")

    dataset = ChessLabeledTensorDataset(states, policies, values)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    # Load models
    print("[INFO] Loading models...")
    model_a = AlphaZeroModel().to(device)
    model_a.load_state_dict(torch.load(model_a_path, map_location=device))
    model_a.eval()

    model_b = AlphaZeroModel().to(device)
    model_b.load_state_dict(torch.load(model_b_path, map_location=device))
    model_b.eval()

    # Evaluate both
    print("\n[Running evaluation...]")
    results_a = evaluate_model(model_a, loader, device)
    results_b = evaluate_model(model_b, loader, device)

    # Print comparison
    print("\n=== Head-to-Head Evaluation ===")
    print(f"{'Metric':<20} {'Ep5':>12} {'Ep10':>12}")
    print("-" * 46)
    for k in results_a.keys():
        print(f"{k:<20} {results_a[k]:>12.4f} {results_b[k]:>12.4f}")
    print("=" * 46)

    if results_b["top1_acc"] > results_a["top1_acc"]:
        print("\n✅ Model B (ep10) outperforms Model A (ep5) on Top-1 accuracy!")
    else:
        print("\n⚖️  Model A (ep5) performs better or equally well.")

if __name__ == "__main__":
    main()