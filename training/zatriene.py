import torch
import numpy as np
from pathlib import Path
from collections import Counter

SELFPLAY_DIR = Path("/home/presi/projects/chess_ai_project/training/training/rl/selfplay_data")

def analyze_episode(path):
    data = torch.load(path, map_location="cpu")
    examples = data.get("examples", [])
    if not examples:
        return None

    # value targets са третият елемент от tuples
    values = [float(ex[2]) for ex in examples if len(ex) >= 3]
    return {
        "file": path.name,
        "n_positions": len(values),
        "mean_value": np.mean(values),
        "value_counts": Counter(values),
    }

def summarize(episodes):
    total_positions = sum(ep["n_positions"] for ep in episodes)
    all_values = np.concatenate([[v] * c for ep in episodes for v, c in ep["value_counts"].items()])
    win_ratio = np.mean(all_values == 1)
    loss_ratio = np.mean(all_values == -1)
    draw_ratio = np.mean(all_values == 0)
    print("\n===== SELF-PLAY DIAGNOSTIC REPORT =====")
    print(f"Total episodes: {len(episodes)}")
    print(f"Total positions: {total_positions}")
    print(f"Win ratio:  {win_ratio:.3f}")
    print(f"Draw ratio: {draw_ratio:.3f}")
    print(f"Loss ratio: {loss_ratio:.3f}")
    print(f"Mean value target: {np.mean(all_values):.3f}")
    print(f"Std value target:  {np.std(all_values):.3f}")
    print("========================================\n")

    for ep in episodes:
        vc = ep['value_counts']
        print(f"{ep['file']:<35}  positions={ep['n_positions']:<5d}  "
              f"mean={ep['mean_value']:.3f}  "
              f"W:{vc.get(1.0,0):<3} D:{vc.get(0.0,0):<3} L:{vc.get(-1.0,0):<3}")
    print("========================================")

def main():
    files = sorted(SELFPLAY_DIR.glob("episode_*.pt"))
    if not files:
        print(f"No selfplay files found in {SELFPLAY_DIR}")
        return

    episodes = []
    for f in files:
        ep = analyze_episode(f)
        if ep:
            episodes.append(ep)
    if episodes:
        summarize(episodes)
    else:
        print("No valid episodes found.")

if __name__ == "__main__":
    main()
