import torch
import numpy as np
from pathlib import Path
from models.AlphaZero import AlphaZeroModel
from training.move_encoding import index_to_move
from torch.nn import functional as F

# --- Конфигурация ---
model_path = Path("C:/Users/prezi/PycharmProjects/chess_ai_project/training/alpha_zero_supervised.pth")
test_dir = Path("C:/Users/prezi/PycharmProjects/chess_ai_project/training/test")

# --- Зареждане на данните ---
def load_multiple_npy(prefix: str, folder: Path):
    files = sorted(folder.glob(f"{prefix}*.npy"))
    arrays = [np.load(f) for f in files]
    return np.concatenate(arrays, axis=0)

states = load_multiple_npy("test_labeled_states_", test_dir)
policies = load_multiple_npy("test_labeled_policies_", test_dir)
values = load_multiple_npy("test_labeled_values_", test_dir)

# --- Подготовка на модел ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlphaZeroModel()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# --- Извличане на произволен пример ---
idx = np.random.randint(len(states))

state_tensor = torch.tensor(states[idx], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
policy_target = torch.tensor(policies[idx], dtype=torch.float32).to(device)
value_target = float(values[idx])

with torch.no_grad():
    policy_pred, value_pred = model(state_tensor)
    policy_pred = policy_pred.squeeze()
    value_pred = value_pred.item()

# --- Обработка на резултатите ---
softmax_policy = torch.softmax(policy_pred, dim=0).cpu().numpy()
target_index = torch.argmax(policy_target).item()
top3_indices = softmax_policy.argsort()[-3:][::-1]

# --- Принтиране на резултатите ---
print("\n--- Prediction Example ---")
print(f"True Value:     {value_target:.3f}")
print(f"Predicted Value:{value_pred:.3f}")
print(f"Target Move:    {index_to_move(target_index)}")
print("Top 3 Predicted Moves:")
for idx in top3_indices:
    print(f"  {str(index_to_move(idx)):<6} → prob={softmax_policy[idx]:.4f}")

