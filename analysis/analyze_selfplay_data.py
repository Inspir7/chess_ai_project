import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import chess
import chess.pgn

# === Directories ===
DEFAULT_DIR = Path(__file__).resolve().parent / "rl" / "selfplay_data"
ALT_DIR = Path("/home/presi/projects/chess_ai_project/training/training/rl/selfplay_data")
SELFPLAY_DIR = DEFAULT_DIR if DEFAULT_DIR.exists() else ALT_DIR

print(f"[INFO] Using self-play data from: {SELFPLAY_DIR}")

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Extract episode stats
# ======================================================================
def extract_episode_stats(pt_file):
    data = torch.load(pt_file, map_location="cpu")
    examples = data.get("examples", [])
    state0 = examples[0][0] if examples else None

    values = []
    entropies = []
    players = []
    game_len = len(examples)

    for ex in examples:
        if len(ex) != 3:
            continue

        state, pi, value = ex

        # VALUE
        v = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
        values.append(v)

        # ENTROPY
        try:
            probs = pi.numpy() if isinstance(pi, torch.Tensor) else np.array(pi)
            ent = -np.sum(probs * np.log(probs + 1e-9))
        except Exception:
            ent = np.nan
        entropies.append(ent)

        # PLAYER
        try:
            turn_plane = state[14]
            white_to_move = (turn_plane[0, 0].item() == 1.0)
            players.append(1 if white_to_move else -1)
        except:
            players.append(0)

    # DETECT OPENING from first moves
    opening_name = None
    try:
        board = chess.Board()
        # Find first played move stored in pi
        if examples:
            pi_vec = examples[0][1]
            idx = np.argmax(pi_vec)
            # decode? Instead: fallback → detect opening from FEN plane  (channel 14 is side)
            # better: inspect first 2-3 moves
        opening_name = "Unknown"
    except:
        opening_name = "Unknown"

    return (
        np.array(values),
        np.array(entropies),
        np.array(players),
        game_len,
        opening_name,
    )


# ======================================================================
# LOAD ALL EPISODES
# ======================================================================
def load_all_episodes(directory):
    files = sorted(directory.glob("episode_*.pt"))
    print(f"[INFO] Found {len(files)} episodes.")

    episodes = []
    for f in files:
        v, e, p, L, opening = extract_episode_stats(f)
        if len(v) == 0:
            print(f"[WARN] Skipping empty {f.name}")
            continue

        episodes.append({
            "file": f.name,
            "values": v,
            "entropy": np.nanmean(e),
            "players": p,
            "length": L,
            "opening": detect_opening_from_examples(v, p, f),
        })

    return episodes


# ======================================================================
# Opening detection from the first move of the episode
# ======================================================================
def detect_opening_from_examples(values, players, pt_file):
    """We derive opening from filename or reconstruct the first move from saved tensor."""
    try:
        data = torch.load(pt_file, map_location="cpu")
        examples = data.get("examples", [])
        if not examples:
            return "Unknown"

        # Reconstruct first board from FEN planes
        # Example format: state_tensor is 15x8x8
        state0 = examples[0][0]
        # Let's guess opening by reading the first move from saved history: we do not store moves,
        # so alternative: derive from board move count if present
        # Simpler fallback: detect opening from first move using pi distribution
        pi0 = examples[0][1]
        idx = int(np.argmax(pi0.numpy()))
        # approximate move classification
        return classify_move_index(idx)

    except Exception as e:
        return "Unknown"


def classify_move_index(idx):
    """Approximate mapping: we classify openings only by general-category."""
    # This is approximate because decode_mapping is not stored.
    # We rely on index ranges:
    if idx < 500:
        return "e4 or similar"
    if 500 <= idx < 1000:
        return "d4 or similar"
    if 1000 <= idx < 1500:
        return "c4 / Nf3 systems"
    return "Other"


# ======================================================================
# ANALYSIS
# ======================================================================
def analyze(episodes):

    # concat everything
    all_values = np.concatenate([ep["values"] for ep in episodes])
    all_players = np.concatenate([ep["players"] for ep in episodes])
    entropies = [ep["entropy"] for ep in episodes]
    lengths = [ep["length"] for ep in episodes]
    openings = [ep["opening"] for ep in episodes]

    # ======== Convert value(sign) -> absolute z (White perspective)
    z_values = all_values * all_players

    wins = np.sum(z_values > 0.5)
    draws = np.sum(np.isclose(z_values, 0.0, atol=1e-3))
    losses = np.sum(z_values < -0.5)
    total = len(z_values)

    print("\n[ABSOLUTE RESULTS]")
    print(f" White wins : {wins} ({wins/total:.1%})")
    print(f" Draws      : {draws} ({draws/total:.1%})")
    print(f" Black wins : {losses} ({losses/total:.1%})")

    # ==================================================================
    # Compute Self-Elo
    # ==================================================================
    per_game_scores = []
    for ep in episodes:
        v = ep["values"]
        p = ep["players"]
        z = v * p

        w = np.sum(z > 0.5)
        d = np.sum(np.isclose(z, 0, atol=1e-3))
        l = np.sum(z < -0.5)

        if w + d + l == 0:
            score = 0.5
        else:
            score = (w + 0.5*d) / (w+d+l)

        per_game_scores.append(score)

    per_game_scores = np.array(per_game_scores)

    # Elo transform
    def score_to_elo(s):
        if s <= 0 or s >= 1:
            return 0
        return 400 * np.log10(s / (1 - s))

    elos = np.array([score_to_elo(s) for s in per_game_scores])

    # Exponential smoothing for stability
    smoothed_elo = []
    alpha = 0.1
    prev = 0
    for e in elos:
        new = alpha*e + (1-alpha)*prev
        smoothed_elo.append(new)
        prev = new

    smoothed_elo = np.array(smoothed_elo)

    print("\n[SELF-ELO]")
    print(f" Final Self-Elo: {smoothed_elo[-1]:.1f}")

    # ==================================================================
    # Opening distribution
    # ==================================================================
    unique, counts = np.unique(openings, return_counts=True)
    opening_dist = sorted(zip(unique, counts), key=lambda x: -x[1])

    print("\n[OPENING DISTRIBUTION — top 10]")
    for op, c in opening_dist[:10]:
        print(f" {op:<20} : {c} games")

    # ==================================================================
    # PLOTS
    # ==================================================================
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PLOTS_DIR / f"rl_diagnostics_{ts}.png"

    plt.figure(figsize=(18, 12))

    # 1) Absolute z histogram
    plt.subplot(2,3,1)
    plt.hist(z_values, bins=[-1.1,-0.5,0.5,1.1],
             rwidth=0.9, color="skyblue")
    plt.xticks([-1,0,1], ["Black win", "Draw", "White win"])
    plt.title("Absolute Results (z-values)")

    # 2) Entropy
    plt.subplot(2,3,2)
    plt.plot(entropies, marker="o")
    plt.title("Entropy per Episode")

    # 3) Game length
    plt.subplot(2,3,3)
    plt.plot(lengths, marker="o")
    plt.title("Game Length per Episode")

    # 4) Self-Elo
    plt.subplot(2,3,4)
    plt.plot(smoothed_elo, marker="o")
    plt.title("Self-Elo (smoothed)")

    # 5) Opening distribution (top 10)
    plt.subplot(2,3,5)
    names = [op for op,_ in opening_dist[:10]]
    vals = [c for _,c in opening_dist[:10]]
    plt.bar(names, vals)
    plt.xticks(rotation=45, ha="right")
    plt.title("Top 10 Openings")

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
