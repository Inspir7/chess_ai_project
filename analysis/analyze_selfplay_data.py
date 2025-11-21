import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime

# === Пътища ===
DEFAULT_DIR = Path(__file__).resolve().parent / "rl" / "selfplay_data"
ALT_DIR = Path("/home/presi/projects/chess_ai_project/training/training/rl/selfplay_data")

SELFPLAY_DIR = DEFAULT_DIR if DEFAULT_DIR.exists() else ALT_DIR
print(f"[INFO] Using self-play data from: {SELFPLAY_DIR}")

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================================
# HELPER: извличане на value + entropy + turn_info
# ======================================================================
def extract_episode_stats(pt_file):
    data = torch.load(pt_file, map_location="cpu")
    examples = data.get("examples", [])

    values = []
    entropies = []
    players = []     # +1 ако white to move, -1 ако black to move

    for ex in examples:
        if len(ex) != 3:
            continue
        state, pi, value = ex

        # ---- value ----
        try:
            v = float(value) if not isinstance(value, torch.Tensor) else value.item()
        except Exception:
            continue
        values.append(v)

        # ---- entropy ----
        try:
            probs = pi.numpy() if isinstance(pi, torch.Tensor) else np.array(pi)
            probs = probs.astype(np.float32)
            ent = -np.sum(probs * np.log(probs + 1e-9))
        except Exception:
            ent = np.nan
        entropies.append(ent)

        # ---- detect whose move it was (white or black) ----
        # твоя input tensor е (15,8,8). Първият канал съдържа състояние на белите?
        # Но ти ползваш fen_to_tensor → там има "side to move" флаг (канал 14)
        # Той е 1 ако е бял на ход, 0 ако е черен на ход.
        try:
            turn_plane = state[14]  # (8x8) плоскост
            white_to_move = (turn_plane[0, 0].item() == 1.0)
            players.append(1 if white_to_move else -1)
        except:
            players.append(0)

    return np.array(values), np.array(entropies), np.array(players)


# ======================================================================
# LOAD ALL EPISODES
# ======================================================================
def load_all_episodes(directory):
    files = sorted(directory.glob("episode_*.pt"))
    print(f"[INFO] Found {len(files)} self-play episodes.\n")

    episodes = []
    for f in files:
        v, e, p = extract_episode_stats(f)
        if len(v) == 0:
            print(f"[WARN] Skipping empty {f.name}")
            continue

        episodes.append({
            "file": f.name,
            "values": v,
            "entropy": np.nanmean(e),
            "players": p
        })

    return episodes


# ======================================================================
# MAIN PLOT / ANALYSIS
# ======================================================================
def analyze(episodes):
    if not episodes:
        print("[ERROR] No episodes to analyze.")
        return

    all_values = np.concatenate([ep["values"] for ep in episodes])
    all_players = np.concatenate([ep["players"] for ep in episodes])
    entropies = [ep["entropy"] for ep in episodes]
    avg_values_per_ep = [np.mean(ep["values"]) for ep in episodes]

    # ============================================================
    # WIN/DRAW/LOSS — perspective-correct
    # ============================================================
    wins = np.sum(all_values > 0.5)
    draws = np.sum(np.isclose(all_values, 0.0, atol=1e-3))
    losses = np.sum(all_values < -0.5)

    total = len(all_values)
    print("\n[GLOBAL VALUE DISTRIBUTION]")
    print(f"  Wins:  {wins} ({wins/total:.1%})")
    print(f"  Draws: {draws} ({draws/total:.1%})")
    print(f"  Loss:  {losses} ({losses/total:.1%})")

    # ============================================================
    # Breakdown by SIDE TO MOVE
    # ============================================================
    mask_white = (all_players == 1)
    mask_black = (all_players == -1)

    def pct(arr):
        return f"{np.sum(arr):d} ({np.sum(arr)/total:.1%})"

    print("\n[VALUE by SIDE TO MOVE]")
    print("White-to-move:")
    print("  +1 wins: ", pct((all_values > 0.5) & mask_white))
    print("  -1 loss: ", pct((all_values < -0.5) & mask_white))
    print("  0 draw: ", pct((np.isclose(all_values, 0.0, atol=1e-3) & mask_white)))

    print("\nBlack-to-move:")
    print("  +1 wins: ", pct((all_values > 0.5) & mask_black))
    print("  -1 loss: ", pct((all_values < -0.5) & mask_black))
    print("  0 draw: ", pct((np.isclose(all_values, 0.0, atol=1e-3) & mask_black)))

    # ============================================================
    # Correlation entropy ↔ average value
    # ============================================================
    valid = [(e, a) for e, a in zip(entropies, avg_values_per_ep) if not np.isnan(e)]
    if valid:
        arr_e = np.array([v[0] for v in valid])
        arr_a = np.array([v[1] for v in valid])
        corr = np.corrcoef(arr_e, arr_a)[0, 1]
    else:
        corr = np.nan

    print(f"\n[CORRELATION] Entropy ↔ Avg Value = {corr:.3f}")

    # ============================================================
    # ======= PLOT =======
    # ============================================================
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PLOTS_DIR / f"rl_diagnostics_{ts}.png"

    plt.figure(figsize=(14, 11))

    # Histogram values
    plt.subplot(2, 2, 1)
    plt.hist(all_values, bins=[-1.1, -0.5, 0.5, 1.1],
             rwidth=0.9, color="skyblue", edgecolor="black")
    plt.xticks([-1, 0, 1], ["Loss", "Draw", "Win"])
    plt.title("Value Distribution")

    # Entropy per episode
    plt.subplot(2, 2, 2)
    plt.plot(entropies, marker="o")
    plt.title("Entropy per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Entropy")

    # Avg values per episode
    plt.subplot(2, 2, 3)
    plt.plot(avg_values_per_ep, marker="o")
    plt.title("Average Value per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Avg Value")

    # ===== NEW: adaptive y-axis zoom =====
    vals = np.array(avg_values_per_ep)

    vmin, vmax = vals.min(), vals.max()

    # ако всичко е 0 → сложи малко пространство
    if abs(vmax - vmin) < 1e-6:
        padding = 0.005
    else:
        padding = (vmax - vmin) * 0.2

    plt.ylim(vmin - padding, vmax + padding)
    # =====================================

    # Scatter entropy vs avg value
    plt.subplot(2, 2, 4)
    plt.scatter(entropies, avg_values_per_ep, s=80, c="purple")
    plt.title(f"Entropy vs Avg Value (corr={corr:.2f})")
    plt.xlabel("Entropy")
    plt.ylabel("Avg Value")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"\n[OK] Plot saved → {out_path}")


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    episodes = load_all_episodes(SELFPLAY_DIR)
    analyze(episodes)
