import torch
import chess
import numpy as np
import sys
import os

# SETUP PATHS
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.AlphaZero import AlphaZeroModel
from training.mcts import MCTS
# Импортираме правилната функция от твоя файл
from training.move_encoding import index_to_move, move_to_index

# CONFIG
MODEL_PATH = os.path.join(PROJECT_ROOT, "training/rl/checkpoints/alpha_zero_rl_checkpoint_final.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def debug_position(fen, description, ai_plays_as):
    print(f"\n==========================================")
    print(f"🔍 TEST: {description}")
    print(f"FEN: {fen}")

    board = chess.Board(fen)
    print(board)

    # 1. Зареждане на модела
    model = AlphaZeroModel().to(DEVICE)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    # 2. Пускаме MCTS (Той използва fen_to_tensor вътрешно)
    # Ако тук се провали, значи проблемът е в MCTS или fen_to_tensor
    print(f"\n🤖 AI thinking as {ai_plays_as}...")

    mcts = MCTS(model, DEVICE, simulations=200, dirichlet_epsilon=0.0)
    pi = mcts.run(board, move_number=50)  # Лъжем за move number, за да няма Temp

    # 3. Анализ на резултата
    print("\n📊 MCTS Top Moves:")
    sorted_moves = sorted(pi.items(), key=lambda x: x[1], reverse=True)[:3]

    if not sorted_moves:
        print("❌ NO MOVES FOUND! (Crash or Illegal moves)")
        return

    found_mate = False
    for mv, prob in sorted_moves:
        print(f"   Move: {mv} | Prob: {prob:.4f}")

        # Проверка дали ходът води до мат
        board.push(mv)
        if board.is_checkmate():
            print("   ✅ MAT FOUND! (AI sees the win)")
            found_mate = True
        board.pop()

    if not found_mate:
        print("   ⚠️ WARNING: AI missed the mate in 1!")


# --- SCENARIOS ---

if __name__ == "__main__":
    # 1. Тест за Белите (Мат с Дама на g7 или h7)
    fen_white = "7k/6pp/7Q/8/8/8/8/6K1 w - - 0 1"
    debug_position(fen_white, "WHITE to move (Queen mate)", "White")

    # 2. Тест за Черните (Мат с Дама на g2 или h2)
    # ТУК Е КРИТИЧНИЯТ ТЕСТ. Ако тук се провали, значи не обръщаме дъската!
    fen_black = "6k1/8/8/8/8/7q/6PP/7K b - - 0 1"
    debug_position(fen_black, "BLACK to move (Queen mate)", "Black")