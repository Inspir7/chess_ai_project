import torch
import chess
import numpy as np
from models.AlphaZero import AlphaZeroModel
from data.generate_labeled_data import determine_phase, move_to_index  # твоите функции

# Устройството
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Зареждаме архитектурата и теглата
model = AlphaZeroModel().to(DEVICE)
model.load_state_dict(torch.load("alpha_zero_supervised.pth", map_location=DEVICE))
model.eval()

def fen_to_tensor(fen):
    """Същият 8×8×15 тензор, който използвахме при обучението."""
    import numpy as np, chess
    board = chess.Board(fen)
    piece_map = { 'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,
                  'p':6,'n':7,'b':8,'r':9,'q':10,'k':11 }
    tensor = np.zeros((15,8,8), dtype=np.float32)
    for sq,p in board.piece_map().items():
        x,y = divmod(sq,8)
        tensor[piece_map[p.symbol()],7-x,y] = 1
    tensor[12,:,:] = 1.0 if board.turn == chess.WHITE else 0.0
    tensor[13,:,:] = board.fullmove_number / 100
    tensor[14,:,:] = determine_phase(board)
    return torch.from_numpy(tensor).unsqueeze(0)

# Примерни тестови позиции
test_fens = [
    "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 2 3",
    "8/8/8/8/8/8/8/8 w - - 0 1"
]

for fen in test_fens:
    board = chess.Board(fen)
    x = fen_to_tensor(fen).to(DEVICE)
    with torch.no_grad():
        policy_logits, value = model(x)
        probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

    # Построй обратен речник index → uci ход само за легалните ходове
    idx_to_move = {}
    for mv in board.legal_moves:
        uci = mv.uci()
        try:
            idx = move_to_index(mv)  # подаваме Move, а не string
            idx_to_move[idx] = uci
        except Exception:
            # ако не успее да намери индекс, просто го пропускаме
            continue

    # Вземаме топ‑5 индекса по вероятност
    top_idxs = np.argsort(probs)[::-1]
    print(f"\nPosition: {fen}")
    print(f"Value prediction: {value.item():.3f}")
    print("Top 5 legal moves and probabilities:")
    cnt = 0
    for idx in top_idxs:
        if idx in idx_to_move:
            print(f"  {idx_to_move[idx]} : {probs[idx]:.4f}")
            cnt += 1
            if cnt == 5:
                break
    if cnt == 0:
        print("  (няма легални ходове или несъответстващи индекси)")
