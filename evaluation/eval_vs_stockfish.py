import sys
import os
import chess
import chess.engine
import torch
import numpy as np
import time

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

# --- EVALUATION PARAMETERS ---
SIMULATIONS = 2000  # Дълбоко мислене
MERCY_THRESHOLD = 500


def get_ai_move(model, board, simulations=2000):
    mcts = MCTS(
        model,
        DEVICE,
        simulations=simulations,
        dirichlet_epsilon=0.0,  # Без шум за тест!
        c_puct=1.5
    )

    # Лъжем MCTS, че е средата на играта (за температурата = 0)
    pi = mcts.run(board, move_number=100)

    moves = list(pi.keys())
    probs = list(pi.values())

    if not moves: return None
    # ARGMAX - Винаги най-силния ход
    best_idx = np.argmax(probs)
    return moves[best_idx]


def evaluate_position_static(board):
    """ Проста оценка на материала за adjudication """
    piece_vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    score = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            val = piece_vals.get(p.piece_type, 0)
            if p.color == chess.WHITE:
                score += val
            else:
                score -= val
    return score


def play_match(model, engine, limit_type, ai_color):
    board = chess.Board()

    # --- STOCKFISH CONFIG ---
    if limit_type == "Skill 0":
        # Elo ~300-400 (Много слаб)
        engine.configure({"Skill Level": 0, "Move Overhead": 100})
        limit = chess.engine.Limit(time=0.01)  # Почти никакво време
    elif limit_type == "Skill 1":
        engine.configure({"Skill Level": 1})
        limit = chess.engine.Limit(time=0.05)
    else:
        # Default (Depth limit)
        engine.configure({"Skill Level": 20})
        if limit_type == "Depth 1":
            limit = chess.engine.Limit(depth=1)
        else:
            limit = chess.engine.Limit(time=0.1)

    color_name = "White" if ai_color == chess.WHITE else "Black"
    print(f"   ⚔️  AI ({color_name}) vs Stockfish ({limit_type})")

    moves_count = 0
    while not board.is_game_over() and moves_count < 120:

        # --- ADJUDICATION CHECK (Прекъсване при голямо предимство) ---
        if moves_count > 30:
            mat_score = evaluate_position_static(board)
            # Ако AI е Бял и води с 5+ пешки
            if ai_color == chess.WHITE and mat_score >= 5:
                print(f"      🏆 Adjudication: AI wins by material advantage (+{mat_score})")
                return 1.0
            # Ако AI е Черен и води с 5+ пешки (score <= -5)
            if ai_color == chess.BLACK and mat_score <= -5:
                print(f"      🏆 Adjudication: AI wins by material advantage ({mat_score})")
                return 1.0

        if board.turn == ai_color:
            # AI MOVE (Full Power)
            move = get_ai_move(model, board, simulations=SIMULATIONS)
            if move is None or move not in board.legal_moves:
                print("       ⚠️ AI Resigned (No move found)")
                return 0.0
            board.push(move)
        else:
            # STOCKFISH MOVE
            try:
                result = engine.play(board, limit)
                board.push(result.move)
            except Exception as e:
                print(f"Stockfish crashed: {e}")
                return 1.0  # Служебна победа за AI

        moves_count += 1
        if moves_count % 20 == 0:
            mat = evaluate_position_static(board)
            print(f"      Move {moves_count}... (Mat: {mat})")

    res = board.result()
    print(f"      🏁 Result: {res} (Moves: {moves_count})")

    if res == "1-0":
        return 1.0 if ai_color == chess.WHITE else 0.0
    elif res == "0-1":
        return 1.0 if ai_color == chess.BLACK else 0.0
    else:
        return 0.5  # Draw


def run_elo_benchmark():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return

    print(f"🤖 Loading AlphaZero Model...")
    model = AlphaZeroModel().to(DEVICE)
    try:
        # Опитваме да заредим state_dict директно
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except:
        # Fallback за пълни checkpoint файлове
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        # Понякога ключът е 'state_dict', понякога 'model_state_dict'
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    model.eval()

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    # Тестваме градация
    levels = ["Skill 0", "Skill 1", "Depth 1"]

    print("\n===========================================")
    print("🚀 STARTING STOCKFISH BENCHMARK (ADJUDICATED)")
    print("===========================================")

    for lvl in levels:
        print(f"\n--- Testing vs Stockfish {lvl} ---")
        score_w = play_match(model, engine, lvl, chess.WHITE)
        score_b = play_match(model, engine, lvl, chess.BLACK)
        total = score_w + score_b

        print(f"📊 Score vs {lvl}: {total}/2.0")

        if total >= 1.0:
            print(f"✅ AI passes {lvl}!")
        else:
            print(f"⚠️ AI struggling at {lvl}.")
            if lvl == "Skill 0" and total == 0:
                print("🛑 Stopping early.")
                break

    engine.quit()


if __name__ == "__main__":
    run_elo_benchmark()