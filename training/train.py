import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.AlphaZero import AlphaZeroModel
from training.move_encoding import flip_move_index


# ============================================================
# DATA AUGMENTATION HELPERS
# ============================================================

def flip_state(tensor_chw: torch.Tensor):
    """
    tensor shape: (15, 8, 8)
    Horizontal mirror (file a <-> file h).
    """
    return torch.flip(tensor_chw, dims=[2])  # flip width


def flip_policy_index(pi_idx: torch.Tensor):
    """
    Takes an integer index (0..4671), applies mirror transform.
    """
    new_idx = flip_move_index(int(pi_idx.item()))
    return torch.tensor(new_idx, dtype=torch.long)


def flip_value(v: torch.Tensor):
    """
    Value DOES NOT change when flipping board left-right.
    (We do not flip perspective)
    """
    return v


# ===================================================================
#  STREAMING DATASET (RAM-freiendly)
# ===================================================================
class ShardedIterableDataset(IterableDataset):
    def __init__(self, folder, prefix, augment=False):
        self.folder = folder
        self.prefix = prefix
        self.augment = augment

        self.state_files = sorted(
            f for f in os.listdir(folder)
            if f.startswith(f"{prefix}_states_") and f.endswith(".npy")
        )
        self.policy_files = sorted(
            f for f in os.listdir(folder)
            if f.startswith(f"{prefix}_policies_") and f.endswith(".npy")
        )
        self.value_files = sorted(
            f for f in os.listdir(folder)
            if f.startswith(f"{prefix}_values_") and f.endswith(".npy")
        )

        assert len(self.state_files) == len(self.policy_files) == len(self.value_files)
        self.num_shards = len(self.state_files)
        print(f"[STREAM] Dataset {prefix}: {self.num_shards} shards found.")

    def __iter__(self):
        for i in range(self.num_shards):

            s_path = os.path.join(self.folder, self.state_files[i])
            p_path = os.path.join(self.folder, self.policy_files[i])
            v_path = os.path.join(self.folder, self.value_files[i])

            states  = np.load(s_path, mmap_mode="r")
            policies = np.load(p_path, mmap_mode="r")
            values  = np.load(v_path, mmap_mode="r")

            for j in range(states.shape[0]):
                # NHWC → CHW
                s = torch.tensor(
                    states[j].transpose(2, 0, 1),
                    dtype=torch.float32
                )

                pi_idx = int(np.argmax(policies[j]))
                p = torch.tensor(pi_idx, dtype=torch.long)
                v = torch.tensor(values[j], dtype=torch.float32)

                # =============================
                # OPTIONAL DATA AUGMENTATION
                # =============================
                if self.augment and np.random.rand() < 0.5:
                    s = flip_state(s)
                    p = flip_policy_index(p)
                    v = flip_value(v)

                yield s, p, v


# ===================================================================
# TRAINING LOOP
# ===================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/presi/projects/chess_ai_project/data")
    parser.add_argument("--save", default="/home/presi/projects/chess_ai_project/training/alpha_zero_supervised.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch",  type=int, default=256)
    parser.add_argument("--lr",     type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir="logs/supervised_streaming")

    # Training uses augmentation; validation does NOT
    train_ds = ShardedIterableDataset(args.data_dir, "train_labeled", augment=True)
    val_ds   = ShardedIterableDataset(args.data_dir, "val_labeled", augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch)

    model = AlphaZeroModel().to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ---------------------------------------------------
    # Learning Rate Scheduler: Cosine decay + warmup
    # ---------------------------------------------------
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-5
    )

    CE  = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()

    best = float("inf")

    print("\n========================================")
    print("     TRAINING STARTED (streaming mode)")
    print("========================================\n")

    # ---------------------------------------------------
    # MAIN TRAIN LOOP
    # ---------------------------------------------------
    for ep in range(1, args.epochs + 1):

        # ---------------- TRAIN ----------------
        model.train()
        t_loss = t_acc = batches = 0

        for s, pi_idx, v in train_loader:
            s, pi_idx, v = s.to(device), pi_idx.to(device), v.to(device)
            opt.zero_grad()

            logits, v_out = model(s)
            v_out = v_out.squeeze()

            loss_p = CE(logits, pi_idx)
            loss_v = MSE(v_out, v)
            loss = loss_p + loss_v

            loss.backward()
            opt.step()

            t_loss += loss.item()
            t_acc  += (logits.argmax(1) == pi_idx).float().mean().item()
            batches += 1

        t_loss /= batches
        t_acc  /= batches

        # Step scheduler AFTER epoch
        scheduler.step()

        # ---------------- VAL ----------------
        model.eval()
        v_loss = v_acc = v_batches = 0

        with torch.no_grad():
            for s, pi_idx, v in val_loader:
                s, pi_idx, v = s.to(device), pi_idx.to(device), v.to(device)

                logits, v_out = model(s)
                v_out = v_out.squeeze()

                loss_p = CE(logits, pi_idx)
                loss_v = MSE(v_out, v)
                loss = loss_p + loss_v

                v_loss += loss.item()
                v_acc  += (logits.argmax(1) == pi_idx).float().mean().item()
                v_batches += 1

        v_loss /= v_batches
        v_acc  /= v_batches

        print(f"[Epoch {ep}] "
              f"Train loss={t_loss:.4f} acc={t_acc:.4f} | "
              f"Val loss={v_loss:.4f} acc={v_acc:.4f} | "
              f"LR={scheduler.get_last_lr()[0]:.6f}")

        writer.add_scalar("train/loss", t_loss, ep)
        writer.add_scalar("train/acc",  t_acc, ep)
        writer.add_scalar("val/loss",   v_loss, ep)
        writer.add_scalar("val/acc",    v_acc, ep)

        if v_loss < best:
            best = v_loss
            torch.save(model.state_dict(), args.save)
            print(f"   ✓ Saved best checkpoint → {args.save}")

    writer.close()


if __name__ == "__main__":
    main()