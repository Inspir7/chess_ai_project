import sys
import os
import torch
from collections import Counter

# project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.AlphaZero import AlphaZeroModel
from training.self_play import play_episode


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlphaZeroModel().to(device)
    ckpt_path = os.path.join(os.path.dirname(__file__), "alpha_zero_supervised.pth")
    print(f"[INFO] Loading supervised model from: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    results = []
    lengths = []

    GAMES = 20
    for i in range(GAMES):
        examples, res, length = play_episode(
            model=model,
            device=device,
            simulations=60,
            base_temperature=1.2,
            verbose=False,
            max_steps=120,
        )
        results.append(res)
        lengths.append(length)
        print(f"Game {i+1:02d}: result={res}, len={length}, positions={len(examples)}")

    c = Counter(results)
    avg_len = sum(lengths) / len(lengths)
    print("\n=== SUMMARY ===")
    print("Results count:", c)
    print(f"Average length: {avg_len:.1f}")


if __name__ == "__main__":
    main()
