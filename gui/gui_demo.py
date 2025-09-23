# file: gui_demo.py
import tkinter as tk
from tkinter import scrolledtext
import torch
import numpy as np
import chess

from models.AlphaZero import AlphaZeroModel
from models.encoder    import Encoder

# --- Подготовка на модела ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder().to(DEVICE)
model   = AlphaZeroModel().to(DEVICE)
model.load_state_dict(torch.load("C:\\Users\prezi\PycharmProjects\chess_ai_project\\training\\alpha_zero_supervised.pth", map_location=DEVICE))
model.eval()

# Повтаряме fen_to_tensor тук, ако е дефинирана някъде другаде, импортирайте я
def fen_to_tensor(fen: str) -> torch.Tensor:
    board = chess.Board(fen)
    piece_map = {
        'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,
        'p':6,'n':7,'b':8,'r':9,'q':10,'k':11
    }
    arr = np.zeros((15,8,8), dtype=np.float32)
    for sq, p in board.piece_map().items():
        i, j = divmod(sq, 8)
        arr[piece_map[p.symbol()], 7-i, j] = 1
    arr[12,:,:] = 1.0 if board.turn == chess.WHITE else 0.0
    arr[13,:,:] = board.fullmove_number / 100
    # детерминиране на фазата (може да импортирате съществуващата функция)
    total = sum({'p':1,'n':3,'b':3,'r':5,'q':9,'k':0}[pc.symbol().lower()]
                for pc in board.piece_map().values())
    phase = 0 if total>20 else 1 if total>10 else 2
    arr[14,:,:] = phase
    return torch.from_numpy(arr).unsqueeze(0).to(DEVICE)

# --- Функция за inference и връщане на текстови резултат ---
def evaluate_fen():
    fen = fen_entry.get().strip()
    try:
        x = fen_to_tensor(fen)
        with torch.no_grad():
            logits, value = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        # избираме само легални ходове:
        board = chess.Board(fen)
        legal = {m.uci(): idx for idx,m in enumerate(board.legal_moves)}
        scored = [(uci, probs[move_to_index(uci)])
                  for uci,move_to_index in legal.items()
                  if 0 <= legal[uci] < 64]
        top5 = sorted(scored, key=lambda x: -x[1])[:5]
        # показваме:
        out.delete('1.0', tk.END)
        out.insert(tk.END, f"Value: {value.item():+.3f}\nTop 5 legal moves:\n")
        for uci, p in top5:
            out.insert(tk.END, f"  {uci} → {p:.3f}\n")
    except Exception as e:
        out.delete('1.0', tk.END)
        out.insert(tk.END, f"Error: {e}")

# --- Построяване на прозореца ---
root = tk.Tk()
root.title("AlphaZero Inference Demo")

tk.Label(root, text="Enter FEN:").pack(padx=10, pady=5)
fen_entry = tk.Entry(root, width=80)
fen_entry.pack(padx=10, pady=5)
fen_entry.insert(0, "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 2 3")

tk.Button(root, text="Evaluate", command=evaluate_fen).pack(pady=10)

out = scrolledtext.ScrolledText(root, width=60, height=10)
out.pack(padx=10, pady=5)

root.mainloop()
