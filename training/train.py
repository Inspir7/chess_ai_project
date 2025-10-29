# train model with the labeled PGN data = supervised

import torch
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.AlphaZero import AlphaZeroModel
from utils.load_numpy_data import load_npy_from_directory
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import io
from PIL import Image

# --- Инициализации ---
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", default="alpha_zero_supervised.pth")
args = parser.parse_args()

writer = SummaryWriter(log_dir="logs/supervised")

# --- Зареждане на данни ---
train_states, train_policies, train_values = load_npy_from_directory("/home/presi/projects/chess_ai_project/training/train")
val_states, val_policies, val_values = load_npy_from_directory("/home/presi/projects/chess_ai_project/training/validation")

train_dataset = TensorDataset(
    torch.tensor(train_states, dtype=torch.float32).permute(0, 3, 1, 2),
    torch.tensor(train_policies, dtype=torch.float32),
    torch.tensor(train_values, dtype=torch.float32)
)

val_dataset = TensorDataset(
    torch.tensor(val_states, dtype=torch.float32).permute(0, 3, 1, 2),
    torch.tensor(val_policies, dtype=torch.float32),
    torch.tensor(val_values, dtype=torch.float32)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# --- Модел ---
model = AlphaZeroModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
policy_loss_fn = nn.CrossEntropyLoss()
value_loss_fn = nn.MSELoss()

# --- Обучение ---
epochs = 10
best_val_loss = float("inf")

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    total_train_acc = 0
    total_train_policy_loss = 0
    total_train_value_loss = 0

    for states, policies, values in train_loader:
        states, policies, values = states.to(device), policies.to(device), values.to(device)

        optimizer.zero_grad()
        policy_out, value_out = model(states)

        policy_loss = policy_loss_fn(policy_out, torch.argmax(policies, dim=1))
        value_loss = value_loss_fn(value_out.squeeze(), values)
        loss = policy_loss + value_loss

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        total_train_policy_loss += policy_loss.item()
        total_train_value_loss += value_loss.item()

        correct = (torch.argmax(policy_out, dim=1) == torch.argmax(policies, dim=1)).float().sum()
        total_train_acc += correct.item() / policies.size(0)

    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_acc = total_train_acc / len(train_loader)
    avg_train_policy_loss = total_train_policy_loss / len(train_loader)
    avg_train_value_loss = total_train_value_loss / len(train_loader)

    # --- Валидация ---
    model.eval()
    total_val_loss = 0
    total_val_acc = 0
    total_val_policy_loss = 0
    total_val_value_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for states, policies, values in val_loader:
            states, policies, values = states.to(device), policies.to(device), values.to(device)

            policy_out, value_out = model(states)

            policy_loss = policy_loss_fn(policy_out, torch.argmax(policies, dim=1))
            value_loss = value_loss_fn(value_out.squeeze(), values)
            loss = policy_loss + value_loss

            total_val_loss += loss.item()
            total_val_policy_loss += policy_loss.item()
            total_val_value_loss += value_loss.item()

            pred_labels = torch.argmax(policy_out, dim=1).cpu().numpy()
            true_labels = torch.argmax(policies, dim=1).cpu().numpy()

            all_preds.extend(pred_labels)
            all_targets.extend(true_labels)

            correct = (pred_labels == true_labels).sum()
            total_val_acc += correct / policies.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_acc = total_val_acc / len(val_loader)
    avg_val_policy_loss = total_val_policy_loss / len(val_loader)
    avg_val_value_loss = total_val_value_loss / len(val_loader)

    # --- Извеждане в терминала ---
    print(f"[Epoch {epoch + 1}]")
    print(f"  Train Loss: {avg_train_loss:.4f} | Policy: {avg_train_policy_loss:.4f} | Value: {avg_train_value_loss:.4f} | Acc: {avg_train_acc:.4f}")
    print(f"  Val   Loss: {avg_val_loss:.4f} | Policy: {avg_val_policy_loss:.4f} | Value: {avg_val_value_loss:.4f} | Acc: {avg_val_acc:.4f}")

    # --- TensorBoard ---
    writer.add_scalar("Loss/Train_Total", avg_train_loss, epoch)
    writer.add_scalar("Loss/Train_Policy", avg_train_policy_loss, epoch)
    writer.add_scalar("Loss/Train_Value", avg_train_value_loss, epoch)
    writer.add_scalar("Accuracy/Train", avg_train_acc, epoch)

    writer.add_scalar("Loss/Val_Total", avg_val_loss, epoch)
    writer.add_scalar("Loss/Val_Policy", avg_val_policy_loss, epoch)
    writer.add_scalar("Loss/Val_Value", avg_val_value_loss, epoch)
    writer.add_scalar("Accuracy/Val", avg_val_acc, epoch)

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_targets, all_preds)
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix (Epoch {epoch + 1})")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = np.array(image)
    writer.add_image("Confusion_Matrix", image, epoch, dataformats='HWC')
    buf.close()
    plt.close(fig)

    # --- Записване на най-добрия модел ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), args.save_path)
        print(f"✅ Записан нов най-добър модел: {args.save_path}")

writer.close()
