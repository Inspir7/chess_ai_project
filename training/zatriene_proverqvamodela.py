import torch
import numpy as np
import os
from models.AlphaZero import AlphaZeroModel
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Път към изображения на фигури (можеш да използваш png за всяка фигура)
piece_symbols = {
    0: "wP", 1: "wN", 2: "wB", 3: "wR", 4: "wQ", 5: "wK",
    6: "bP", 7: "bN", 8: "bB", 9: "bR", 10: "bQ", 11: "bK"
}
piece_colors = {
    "w": "white",
    "b": "black"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥️ Device:", device)

# Зареждане на модела
model = AlphaZeroModel().to(device)
model.load_state_dict(torch.load("/home/presi/projects/chess_ai_project/training/alpha_zero_supervised.pth", map_location=device))
model.eval()
print("✅ Моделът е успешно зареден!")

base_dir = "/home/presi/projects/chess_ai_project/training"

def compute_average_loss(split):
    states_files = sorted([f for f in os.listdir(os.path.join(base_dir, split)) if "states" in f])
    policies_files = sorted([f for f in os.listdir(os.path.join(base_dir, split)) if "policies" in f])
    values_files = sorted([f for f in os.listdir(os.path.join(base_dir, split)) if "values" in f])

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_samples = 0

    for s_file, p_file, v_file in zip(states_files, policies_files, values_files):
        states = np.load(os.path.join(base_dir, split, s_file))
        policies = np.load(os.path.join(base_dir, split, p_file))
        values = np.load(os.path.join(base_dir, split, v_file))

        # Преобразуване към torch
        states = torch.tensor(states, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        policies = torch.tensor(policies, dtype=torch.float32, device=device)
        values = torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(1)

        with torch.no_grad():
            pred_policy, pred_value = model(states)

        batch_size = states.shape[0]
        policy_loss = torch.nn.functional.cross_entropy(pred_policy, torch.argmax(policies, dim=1), reduction='sum')
        value_loss = torch.nn.functional.mse_loss(pred_value, values, reduction='sum')

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_samples += batch_size

    avg_policy_loss = total_policy_loss / total_samples
    avg_value_loss = total_value_loss / total_samples

    print(f"📊 [{split.upper()}] Average Policy loss: {avg_policy_loss:.4f}")
    print(f"📊 [{split.upper()}] Average Value loss: {avg_value_loss:.4f}")

def draw_board_with_heatmap(board_tensor, policy_probs, top_k=5):
    """
    board_tensor: 8x8x15 numpy array
    policy_probs: 4672 torch tensor
    """
    heatmap = np.zeros((8,8))
    top_indices = torch.argsort(policy_probs, descending=True)[:top_k]
    for idx in top_indices:
        idx = int(idx)  # <<< конвертираме в int
        from_sq = idx // 64
        to_sq = idx % 64
        fx, fy = divmod(from_sq, 8)
        tx, ty = divmod(to_sq, 8)
        heatmap[7 - tx, ty] += policy_probs[idx].item()

    fig, ax = plt.subplots(figsize=(6,6))
    # Рисуваме шахматното поле
    colors = ["#F0D9B5", "#B58863"]
    for i in range(8):
        for j in range(8):
            rect = patches.Rectangle((j, i), 1, 1, facecolor=colors[(i+j)%2])
            ax.add_patch(rect)

    # Overlay на heatmap
    ax.imshow(heatmap, cmap="hot", alpha=0.6, extent=[0,8,0,8], origin='lower')

    # Добавяме фигури
    piece_map = board_tensor[:,:,:12]  # канали 0-11
    for ch in range(12):
        positions = np.argwhere(piece_map[:,:,ch]==1)
        for pos in positions:
            x, y = pos[1]+0.5, 7 - pos[0]+0.5
            symbol = piece_symbols[ch]
            ax.text(x-0.5+0.5, y-0.5+0.5, symbol, fontsize=20, ha='center', va='center')

    ax.set_xlim(0,8)
    ax.set_ylim(0,8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

# Изчисляване на загуби
for split in ["train", "validation", "test"]:
    compute_average_loss(split)

# Визуализация на няколко примера
states_files = sorted([f for f in os.listdir(os.path.join(base_dir, "train")) if "states" in f])
policies_files = sorted([f for f in os.listdir(os.path.join(base_dir, "train")) if "policies" in f])

states = np.load(os.path.join(base_dir, "train", states_files[0]))
policies = np.load(os.path.join(base_dir, "train", policies_files[0]))

states_tensor = torch.tensor(states, dtype=torch.float32, device=device).permute(0,3,1,2)
policies_tensor = torch.tensor(policies, dtype=torch.float32, device=device)

with torch.no_grad():
    pred_policy, _ = model(states_tensor)

for i in range(3):
    draw_board_with_heatmap(states_tensor[i].permute(1,2,0).cpu().numpy(), pred_policy[i])
