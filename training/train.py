import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from models.AlphaZero import AlphaZeroModel
from training.move_encoding import get_total_move_count

def load_npz(train_path, val_path):
    train = np.load(train_path)
    val   = np.load(val_path)

    return (
        train["states"], train["policies"], train["values"],
        val["states"],   val["policies"],   val["values"]
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="/home/presi/projects/chess_ai_project/data/supervised/supervised_train.npz")
    parser.add_argument("--val",   default="/home/presi/projects/chess_ai_project/data/supervised/supervised_val.npz")
    parser.add_argument("--save",  default="/home/presi/projects/chess_ai_project/training/alpha_zero_supervised.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr",     type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir="logs/supervised_new")

    print("[INFO] Loading dataset…")
    train_states, train_pi, train_v, val_states, val_pi, val_v = load_npz(args.train, args.val)

    train_dataset = TensorDataset(
        torch.tensor(train_states, dtype=torch.float32),
        torch.tensor(train_pi,     dtype=torch.long),
        torch.tensor(train_v,      dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(val_states, dtype=torch.float32),
        torch.tensor(val_pi,     dtype=torch.long),
        torch.tensor(val_v,      dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

    model = AlphaZeroModel().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    CE  = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()

    best = float("inf")

    for ep in range(1, args.epochs + 1):
        # ------------------- TRAIN -------------------
        model.train()
        t_loss = t_acc = 0
        t_batches = 0

        for s, pi_idx, v in train_loader:
            s, pi_idx, v = s.to(device), pi_idx.to(device), v.to(device)
            opt.zero_grad()

            logits, v_out = model(s)
            if v_out.ndim == 2:
                v_out = v_out.squeeze(1)

            loss_p = CE(logits, pi_idx)
            loss_v = MSE(v_out, v)
            loss = loss_p + loss_v

            loss.backward()
            opt.step()

            t_loss += loss.item()
            t_acc += (logits.argmax(1) == pi_idx).float().mean().item()
            t_batches += 1

        t_loss /= t_batches
        t_acc  /= t_batches

        # ------------------- VAL -------------------
        model.eval()
        v_loss = v_acc = 0
        v_batches = 0

        with torch.no_grad():
            for s, pi_idx, v in val_loader:
                s, pi_idx, v = s.to(device), pi_idx.to(device), v.to(device)

                logits, v_out = model(s)
                if v_out.ndim == 2:
                    v_out = v_out.squeeze(1)

                loss_p = CE(logits, pi_idx)
                loss_v = MSE(v_out, v)
                loss = loss_p + loss_v

                v_loss += loss.item()
                v_acc  += (logits.argmax(1) == pi_idx).float().mean().item()
                v_batches += 1

        v_loss /= v_batches
        v_acc  /= v_batches

        print(f"[Epoch {ep}] Train loss={t_loss:.4f} acc={t_acc:.4f} | Val loss={v_loss:.4f} acc={v_acc:.4f}")

        writer.add_scalar("train/loss", t_loss, ep)
        writer.add_scalar("train/acc", t_acc, ep)
        writer.add_scalar("val/loss",   v_loss, ep)
        writer.add_scalar("val/acc",    v_acc, ep)

        if v_loss < best:
            best = v_loss
            torch.save(model.state_dict(), args.save)
            print(f"  ✓ Saved best model → {args.save}")

    writer.close()

if __name__ == "__main__":
    main()