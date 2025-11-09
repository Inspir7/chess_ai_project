#Сравнява предсказаните policy/value изходи на модела спрямо етикетирани данни (supervised test set).
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch.nn.functional as F

from models.AlphaZero import AlphaZeroModel
from training.move_encoding import index_to_move


class ChessLabeledTensorDataset(Dataset):
    def __init__(self, states, policies, values):
        # Вече приемаме готови NumPy масиви
        self.states = states
        self.policies = policies
        self.values = values

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


def evaluate_model(model, dataloader, device):
    total_value_loss = 0.0
    total_policy_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    total = 0

    with torch.no_grad():
        for state, policy_target, value_target in dataloader:
            state = state.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)

            policy_pred, value_pred = model(state)

            value_loss = F.mse_loss(value_pred.squeeze(), value_target.squeeze(), reduction='sum')
            total_value_loss += value_loss.item()

            policy_pred_log = torch.log_softmax(policy_pred, dim=1)
            policy_loss = -(policy_target * policy_pred_log).sum()
            total_policy_loss += policy_loss.item()

            pred_topk = torch.topk(policy_pred, k=3, dim=1).indices
            target_indices = policy_target.argmax(dim=1)

            correct_top1 += (pred_topk[:, 0] == target_indices).sum().item()
            correct_top3 += sum([target_indices[i].item() in pred_topk[i] for i in range(len(target_indices))])
            total += state.size(0)

    avg_value_loss = total_value_loss / total
    avg_policy_loss = total_policy_loss / total
    top1_acc = correct_top1 / total * 100
    top3_acc = correct_top3 / total * 100

    print("\n===== Evaluation Results =====")
    print(f"Value Loss (MSE):       {avg_value_loss:.4f}")
    print(f"Policy Loss (CrossEnt): {avg_policy_loss:.4f}")
    print(f"Top-1 Move Accuracy:    {top1_acc:.2f}%")
    print(f"Top-3 Move Accuracy:    {top3_acc:.2f}%")
    print("==============================\n")


def load_and_concatenate_npy(dir_path: Path, prefix: str):
    files = sorted(dir_path.glob(f"{prefix}_*.npy"))
    arrays = [np.load(f) for f in files]
    return np.concatenate(arrays, axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path("C:/Users/prezi/PycharmProjects/chess_ai_project/training/alpha_zero_supervised.pth")
    test_dir = Path("C:/Users/prezi/PycharmProjects/chess_ai_project/training/test")

    print("[Loading Test Data]")
    states = load_and_concatenate_npy(test_dir, "test_labeled_states")
    policies = load_and_concatenate_npy(test_dir, "test_labeled_policies")
    values = load_and_concatenate_npy(test_dir, "test_labeled_values")

    print(f"Loaded shapes: states={states.shape}, policies={policies.shape}, values={values.shape}")

    dataset = ChessLabeledTensorDataset(states, policies, values)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    model = load_model(model_path, device)
    evaluate_model(model, dataloader, device)


if __name__ == "__main__":
    main()
