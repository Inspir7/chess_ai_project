import sys
import os
import chess
import chess.engine
import torch
import numpy as np
import math

# ==========================
# SETUP PATHS
# ==========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.AlphaZero import AlphaZeroModel
from training.mcts import MCTS

# ==========================
# CONFIG
# ==========================
MODEL_PATH = os.path.join(PROJECT_ROOT, "training/rl/checkpoints/alpha_zero_rl_checkpoint_final.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STOCKFISH_PATH = "/usr/games/stockfish"


def get_ai_move(model, board, simulations=800):
    mcts = MCTS(
        model,
        DEVICE,
        simulations=simulations,
        dirichlet_epsilon=0.0,
        c_puct=1.5
    )

    # Лъжем MCTS, че е средата на играта
    pi = mcts.run(board, move_number=100)

    moves = list(pi.keys())
    probs = list(pi.values())

    if not moves: return None
    best_idx = np.argmax(probs)
    return moves[best_idx]


def play_match(model, engine, limit_type, ai_color):
    board = chess.Board()

    # Тук вече не ползваме Elo настройките, а подаваме лимита директно при хода
    # engine.configure(...) - не е нужно за depth limit

    color_name = "White" if ai_color == chess.WHITE else "Black"
    print(f"   ⚔️  AI ({color_name}) vs Stockfish ({limit_type})")

    moves_count = 0
    while not board.is_game_over() and moves_count < 150:
        if board.turn == ai_color:
            # AI
            move = get_ai_move(model, board, simulations=400)
            if move is None or move not in board.legal_moves:
                return 0.0  # Loss
            board.push(move)
        else:
            # STOCKFISH - СИЛАТА СЕ ОПРЕДЕЛЯ ТУК
            if limit_type == "Depth 1":
                limit = chess.engine.Limit(depth=1)
            elif limit_type == "Depth 2":
                limit = chess.engine.Limit(depth=2)
            else:
                limit = chess.engine.Limit(time=0.1)  # Skill 0 standard

            result = engine.play(board, limit)
            board.push(result.move)

        moves_count += 1
        if moves_count % 30 == 0: print(f"      Move {moves_count}...")

    res = board.result()
    print(f"      🏁 Result: {res} (Moves: {moves_count})")

    if res == "1-0":
        return 1.0 if ai_color == chess.WHITE else 0.0
    elif res == "0-1":
        return 1.0 if ai_color == chess.BLACK else 0.0
    else:
        return 0.5


def run_elo_benchmark():
    if not os.path.exists(MODEL_PATH):
        print("Model not found");
        return

    print(f"🤖 Loading AlphaZero Model...")
    model = AlphaZeroModel().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    # ТЕСТВАМЕ ПО ДЪЛБОЧИНА (Depth) вместо ELO
    # Depth 1 ~= 400-600 ELO
    # Depth 2 ~= 800-1000 ELO
    levels = ["Depth 1", "Depth 2"]

    print("\n===========================================")
    print("🚀 STARTING STOCKFISH DEPTH BENCHMARK")
    print("===========================================")

    for lvl in levels:
        print(f"\n--- Testing vs Stockfish {lvl} ---")

        score_w = play_match(model, engine, lvl, chess.WHITE)
        score_b = play_match(model, engine, lvl, chess.BLACK)

        total = score_w + score_b
        print(f"📊 Score vs {lvl}: {total}/2.0")

        if total > 0.5:
            print(f"✅ AI can compete at {lvl}!")
        else:
            print(f"⚠️ AI struggling at {lvl}.")
            break

    engine.quit()


if __name__ == "__main__":
    run_elo_benchmark()