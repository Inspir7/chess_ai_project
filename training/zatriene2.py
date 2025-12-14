import chess
import torch
import numpy as np

from models.AlphaZero import AlphaZeroModel
from training.move_encoding import move_to_index, get_total_move_count
from data.generate_labeled_data import fen_to_tensor
from training.mcts import MCTS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/presi/projects/chess_ai_project/training/rl/checkpoints/alpha_zero_rl_checkpoint_ep130.pth"

# -----------------------
# Convert board → model input
# -----------------------
def board_to_input(board: chess.Board):
    tens = fen_to_tensor(board.fen())
    arr = np.array(tens, dtype=np.float32)
    if arr.shape == (8, 8, 15):
        arr = arr.transpose(2, 0, 1)
    else:
        arr = arr.reshape(15, 8, 8)
    return arr  # (15,8,8)



def softmax_np(x: np.ndarray):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


# -----------------------
# Load model
# -----------------------
model = AlphaZeroModel().to(DEVICE)
sd = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(sd)
model.eval()

board = chess.Board()

# -----------------------
# RAW MODEL POLICY + VALUE
# -----------------------
with torch.no_grad():
    x = board_to_input(board)
    inp = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1,C,8,8)

    pol_logits, value = model(inp)
    value = float(value.view(-1).item())  # safe scalar

    # logits -> numpy
    logits = pol_logits[0].detach().cpu().numpy().astype(np.float64)  # (4672,)
    pol = softmax_np(logits)

print("=== SANITY: INPUT ===")
print("input shape:", tuple(inp.shape), "dtype:", inp.dtype, "device:", inp.device)
print("input min/max:", float(inp.min().item()), float(inp.max().item()))
print()

print("=== MODEL VALUE HEAD ===")
print(f"Value estimate: {value:.6f}")
print()

# 1) Top indices overall (без да гледаме легалност)
topk = 20
top_idx = np.argsort(-pol)[:topk]
print("=== TOP POLICY INDICES OVERALL ===")
for i in top_idx[:10]:
    print(f"idx={int(i):4d}  p={pol[i]:.8e}")
print()

# 2) Legal moves mass + top legal moves
move_probs = {}
bad_moves = []
for mv in board.legal_moves:
    try:
        idx = move_to_index(mv)
        if not (0 <= idx < len(pol)):
            bad_moves.append((mv, idx, "OUT_OF_RANGE"))
            continue
        move_probs[mv] = float(pol[idx])
    except Exception as e:
        bad_moves.append((mv, None, f"EXC: {e}"))

legal_mass = sum(move_probs.values())
print("=== LEGAL MOVE MASS CHECK ===")
print(f"Legal moves count: {board.legal_moves.count()}")
print(f"Mass on legal moves: {legal_mass:.6f}")
print(f"Mass on NON-legal indices: {1.0 - legal_mass:.6f}")
print()

if bad_moves:
    print("=== MOVE_TO_INDEX PROBLEMS (first 10) ===")
    for item in bad_moves[:10]:
        print(item)
    print()

print("=== TOP MODEL POLICY MOVES (LEGAL) ===")
for mv, p in sorted(move_probs.items(), key=lambda x: -x[1])[:10]:
    print(f"{mv.uci()}: {p:.8e}")
print()

# 3) Entropy (ако е много ниска/висока може да показва колапс)
entropy = -float(np.sum(pol * np.log(pol + 1e-12)))
print("=== POLICY STATS ===")
print(f"Entropy: {entropy:.4f}")
print(f"Max prob: {float(pol.max()):.8e} at idx={int(pol.argmax())}")
print()

# -----------------------
# MCTS
# -----------------------
mcts = MCTS(model, DEVICE, simulations=600)
pi = mcts.run(board, move_number=0)

print("=== MCTS POLICY (top moves) ===")
for mv, p in sorted(pi.items(), key=lambda x: -x[1])[:10]:
    # mv може да е chess.Move или uci-string според твоя MCTS; поддържаме и двете
    u = mv.uci() if hasattr(mv, "uci") else str(mv)
    print(f"{u}: {p:.6f}")
print()

# ============================================================
# POLICY ALIGNMENT: model policy vs MCTS π
# ============================================================

total_moves = get_total_move_count()

# π от MCTS → вектор 4672
pi_vec = np.zeros(total_moves, dtype=np.float64)
for mv, p in pi.items():
    idx = move_to_index(mv)
    if 0 <= idx < total_moves:
        pi_vec[idx] = p

# нормализация (за всеки случай)
if pi_vec.sum() > 0:
    pi_vec /= pi_vec.sum()

# policy от модела вече я имаме: pol (softmax върху логитите)
model_vec = pol.astype(np.float64)

eps = 1e-12

# Cross-Entropy H(π, model)
cross_entropy = -np.sum(pi_vec * np.log(model_vec + eps))

# KL divergence KL(π || model)
kl_div = np.sum(pi_vec * np.log((pi_vec + eps) / (model_vec + eps)))

print("=== POLICY ALIGNMENT ===")
print(f"Cross-Entropy (π , model): {cross_entropy:.4f}")
print(f"KL(π || model):           {kl_div:.4f}")
print()

# -----------------------
# Top move comparison
# -----------------------
print("Top moves comparison:")
print("MCTS π top 5:")
for mv, p in sorted(pi.items(), key=lambda x: -x[1])[:5]:
    print(f"  {mv.uci()}: {p:.3f}")

print("\nModel policy top 5 (legal only):")
for mv, p in sorted(move_probs.items(), key=lambda x: -x[1])[:5]:
    print(f"  {mv.uci()}: {p:.3f}")

print()