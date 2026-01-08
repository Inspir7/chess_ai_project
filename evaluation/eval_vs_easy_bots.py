import random
import chess
import torch
import numpy as np

from config.paths import CHECKPOINTS_DIR
from pathlib import Path

# ==========================
# SETUP PATHS
# ==========================


from models.AlphaZero import AlphaZeroModel
from training.mcts import MCTS

# ==========================
# CONFIG
# ==========================
# Използваме релативен път, за да е по-сигурно
MODEL_PATH = CHECKPOINTS_DIR / "alpha_zero_rl_checkpoint_final.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_ai_move(model, board, simulations=800):
    """
    Връща най-добрия ход според AlphaZero.
    ВАЖНО: Тук изключваме шума и случайността!
    """
    # 1. Създаваме MCTS с epsilon=0.0 (БЕЗ ШУМ!)
    mcts = MCTS(
        model,
        DEVICE,
        simulations=simulations,
        dirichlet_epsilon=0.0,  # <-- ИЗКЛЮЧВАМЕ ШУМА
        c_puct=1.5  # Стандартно за игра
    )

    # 2. Пускаме MCTS
    # move_number=100 лъже MCTS, че не сме в дебюта
    pi = mcts.run(board, move_number=100)

    # 3. GREEDY SELECTION (Най-висока вероятност)
    moves = list(pi.keys())
    probs = list(pi.values())

    if not moves:
        return None

    best_idx = np.argmax(probs)
    best_move = moves[best_idx]

    return best_move


def get_random_move(board):
    return random.choice(list(board.legal_moves))


def print_board(board):
    print("--------------------------------------------------")
    print(board)
    print("--------------------------------------------------")


def play_debug_game(model, ai_color):
    board = chess.Board()
    ai_color_str = "White" if ai_color == chess.WHITE else "Black"

    print(f"\n=== Game (AI is {ai_color_str}) vs RANDOM BOT ===")

    moves_count = 0
    # УВЕЛИЧАВАМЕ симулациите на 800 - дай му време да мисли!
    AI_SIMS = 800

    while not board.is_game_over() and moves_count < 150:
        # --- MERCY RULE (Милост) ---
        # Ако AI води с много материал, признаваме победата веднага
        piece_vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        w_mat = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in piece_vals.items())
        b_mat = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in piece_vals.items())

        if ai_color == chess.WHITE:
            diff = w_mat - b_mat
        else:
            diff = b_mat - w_mat

        if diff >= 10:  # Ако води с 10 точки (Дама + пешка)
            print(f"✅ MERCY RULE: AI води с {diff} точки! Присъждаме победа.")
            break
        # ---------------------------

        if board.turn == ai_color:
            try:
                move = get_ai_move(model, board, simulations=AI_SIMS)
                if move is None or move not in board.legal_moves:
                    print(f"❌ AI TRIED ILLEGAL MOVE: {move}")
                    break
            except Exception as e:
                print(f"❌ AI CRASHED: {e}")
                break
        else:
            move = get_random_move(board)

        board.push(move)
        moves_count += 1

        if moves_count % 10 == 0:
            print(f"Move {moves_count} | Material Diff: {diff}")

    print(f"--- FINAL BOARD (Move {moves_count}) ---")
    print_board(board)
    print(f"Result: {board.result()}")

    # Финална проверка на материала
    piece_vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    w_mat = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in piece_vals.items())
    b_mat = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in piece_vals.items())

    final_diff = w_mat - b_mat if ai_color == chess.WHITE else b_mat - w_mat

    print(f"AI Material Advantage: {final_diff} points")

    if final_diff > 3:
        print(f"✅ РЕЗУЛТАТ: AI печели по материал (+{final_diff})!")
    elif final_diff < -3:
        print(f"❌ РЕЗУЛТАТ: AI загуби материал ({final_diff}).")
    else:
        print(f"⚠️ РЕЗУЛТАТ: Равностойно ({final_diff}).")


if __name__ == "__main__":
    print(f"Loading RL Model: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    model = AlphaZeroModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Играем 2 игри
    play_debug_game(model, chess.WHITE)
    play_debug_game(model, chess.BLACK)