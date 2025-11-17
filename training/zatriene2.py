import torch
import chess

# === Импорти от проекта ===
from models.AlphaZero import AlphaZeroModel
from models.move_encoding import move_to_index, get_total_move_count
from utils.chess_utils import fen_to_tensor


# === НАСТРОЙКИ ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("\n==========================")
print("  TEST 1: MODEL OUTPUT DIM")
print("==========================")

model = AlphaZeroModel().to(DEVICE)
x_dummy = torch.zeros((1, 15, 8, 8), dtype=torch.float32).to(DEVICE)

with torch.no_grad():
    logits, value = model(x_dummy)

print("Model policy output dim =", logits.shape[1])

try:
    move_count = get_total_move_count()
except Exception:
    move_count = None

print("get_total_move_count() =", move_count)

if move_count is not None and logits.shape[1] != move_count:
    print("\n❌ PROBLEM: Model output dim does NOT match move encoding size!")
else:
    print("✓ OK: Model output matches move encoding.")


print("\n==========================")
print("  TEST 2: MOVE ENCODING (Start Position)")
print("==========================")

board = chess.Board()
legal = list(board.legal_moves)
unmapped = []
oob = []

move_count = get_total_move_count() or logits.shape[1]

for mv in legal:
    idx = move_to_index(mv, board)
    if idx is None:
        unmapped.append(mv)
    elif idx < 0 or idx >= move_count:
        oob.append((mv, idx))

print("Total legal moves:", len(legal))
print("Unmapped moves:", len(unmapped))
print("Out-of-bounds mapping:", len(oob))

if unmapped[:10]:
    print("Examples unmapped:", unmapped[:10])
if oob[:10]:
    print("Examples out-of-range:", oob[:10])


print("\n==========================")
print("  TEST 3: TENSOR SHAPE")
print("==========================")

fen = board.fen()
np_tensor = fen_to_tensor(fen)

print("fen_to_tensor shape:", np_tensor.shape)

tensor = torch.tensor(np_tensor).permute(2,0,1).unsqueeze(0)
print("After permute + batch:", tensor.shape)

if tensor.shape != torch.Size([1, 15, 8, 8]):
    print("❌ PROBLEM: Expected shape (1,15,8,8)")
else:
    print("✓ OK: Tensor is correct CHW format.")


print("\n==========================")
print("  TEST 4: FORWARD + BACKWARD")
print("==========================")

state = torch.zeros((1, 15, 8, 8), dtype=torch.float32).to(DEVICE)
policy_target = torch.zeros((1, logits.shape[1]), dtype=torch.float32).to(DEVICE)
policy_target[0, 0] = 1.0  # sample one-hot
value_target = torch.tensor([[0.5]], dtype=torch.float32).to(DEVICE)

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

params_before = sum(p.sum().item() for p in model.parameters())

logits, val = model(state)
loss_p = torch.nn.CrossEntropyLoss()(logits, torch.argmax(policy_target, dim=1))
loss_v = torch.nn.MSELoss()(val, value_target)
loss = loss_p + loss_v

optimizer.zero_grad()
loss.backward()

total_grad_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_grad_norm += p.grad.data.norm().item()

optimizer.step()

params_after = sum(p.sum().item() for p in model.parameters())

print("loss =", loss.item())
print("total_grad_norm =", total_grad_norm)
print("parameter_delta =", params_after - params_before)

if total_grad_norm == 0:
    print("❌ PROBLEM: No gradients — model is not learning!")
else:
    print("✓ OK: Gradients flow.")

if abs(params_after - params_before) < 1e-6:
    print("⚠ WARNING: Parameters barely changed — LR too small or silent error.")
else:
    print("✓ OK: Parameters updated normally.")


print("\n==========================")
print("      ALL TESTS DONE")
print("==========================\n")
