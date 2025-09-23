import os
import torch
from datetime import datetime
from training.self_play import play_episode
from models.AlphaZero import AlphaZeroModel
from training.train_on_batch import train_on_batch
from inference import fen_to_tensor  # Използваме съществуващата функция
from training.mcts import MCTS  # Добавен MCTS агент

# === Настройки ===
base_dir = "/home/presi/projects/chess_ai_project"
save_dir = os.path.join(base_dir, "training", "rl")
model_path = os.path.join(base_dir, "training", "alpha_zero_supervised.pth")
log_file = os.path.join(save_dir, "training_log.txt")
num_episodes = 10

# === Setup ===
os.makedirs(save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlphaZeroModel().to(device)

print(device)
# === Зареждане на предишен модел (ако има) ===
if os.path.exists(model_path):
    print("[RL] Loading previous model...")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("[RL] No previous model found. Starting fresh.")

# === Обучение чрез self-play ===
for i in range(num_episodes):
    print(f"\n[Self-Play] Starting episode {i + 1}/{num_episodes}")

    try:
        episode_data = play_episode(model, device)
    except Exception as e:
        print(f"[RL] ❌ Грешка при епизод {i + 1}: {e}")
        continue

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(save_dir, f"episode_{i + 1}_{timestamp}.pt")
    torch.save(episode_data, out_file)
    print(f"[RL] ✅ Записан епизод: {out_file}")

    try:
        loss_policy, loss_value = train_on_batch(model, episode_data, device)
    except Exception as e:
        print(f"[RL] ❌ Грешка при тренировка: {e}")
        continue

    # Записване на статистика
    with open(log_file, "a") as f:
        f.write(f"{timestamp}, Episode {i + 1}, Loss Policy: {loss_policy:.4f}, Loss Value: {loss_value:.4f}\n")

# === Запис на последния модел ===
torch.save(model.state_dict(), model_path)
print(f"[RL] 💾 Моделът е запазен в: {model_path}")


# === 🔁 Обучение от replay буфер ===
def train_model_from_replay(model, replay_buffer, device):
    import random

    print("[RL] 🔁 Обучение от replay буфер...")
    model.train()

    recent_samples = replay_buffer[-100:] if len(replay_buffer) >= 100 else replay_buffer

    inputs = []
    policy_targets = []
    value_targets = []

    for fen, result in recent_samples:
        board_tensor = fen_to_tensor(fen)

        # Използваме MCTS за извличане на policy
        mcts = MCTS(model=model, device=device)
        mcts_policy = mcts.get_policy(fen)

        value = torch.tensor([result], dtype=torch.float32)

        inputs.append(board_tensor)
        policy_targets.append(mcts_policy)
        value_targets.append(value)

    inputs = torch.stack(inputs).to(device)
    policy_targets = torch.stack(policy_targets).to(device)
    value_targets = torch.stack(value_targets).to(device)

    loss_policy, loss_value = train_on_batch(model, (inputs, policy_targets, value_targets), device)
    print(f"[RL] ✅ Тренирано с replay буфер | Policy Loss: {loss_policy:.4f}, Value Loss: {loss_value:.4f}")
