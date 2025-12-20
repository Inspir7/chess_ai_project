import sys
import os
import random
import chess
import torch
import numpy as np

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

# Стойности на фигурите за "Material Bot"
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 1000  # Царят е безценен
}


def get_material_score(board, color):
    """ Изчислява материалния баланс на дъската """
    score = 0
    for pt, val in PIECE_VALUES.items():
        score += len(board.pieces(pt, color)) * val
        score -= len(board.pieces(pt, not color)) * val
    return score


def get_weak_bot_move(board):
    """
    Minimax Bot (Depth 1):
    - Гледа всички възможни ходове.
    - Избира този, който води до най-добър материален баланс веднага.
    - Ако има няколко еднакви, избира случаен (за разнообразие).
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    best_moves = []
    best_score = -float('inf')

    turn = board.turn  # True for White, False for Black

    for move in legal_moves:
        board.push(move)

        # Оценка от гледна точка на този, който е на ход (turn)
        # Ако е Бял, искаме (White - Black) да е макс.
        # Ако е Черен, искаме (Black - White) да е макс.
        score = get_material_score(board, turn)

        # Ако мат - това е най-доброто!
        if board.is_checkmate():
            score = 9999

        board.pop()

        if score > best_score:
            best_score = score
            best_moves = [move]
        elif score == best_score:
            best_moves.append(move)

    return random.choice(best_moves)


def get_ai_move(model, board, simulations=800):
    mcts = MCTS(model, DEVICE, simulations=simulations, dirichlet_epsilon=0.0, c_puct=1.5)
    try:
        pi = mcts.run(board, move_number=100)
        moves = list(pi.keys())
        probs = list(pi.values())
        if not moves: return None
        return moves[np.argmax(probs)]
    except:
        return None


def play_game(model, ai_color):
    board = chess.Board()
    ai_color_str = "White" if ai_color == chess.WHITE else "Black"
    opponent_str = "Material Bot (Elo ~350)"

    print(f"\n⚔️  AI ({ai_color_str}) vs {opponent_str}")

    moves = 0
    while not board.is_game_over() and moves < 150:
        if board.turn == ai_color:
            # AI
            move = get_ai_move(model, board, simulations=400)
        else:
            # Material Bot
            move = get_weak_bot_move(board)

        if move is None or move not in board.legal_moves:
            print("❌ Illegal move or crash.")
            break

        board.push(move)
        moves += 1

        if moves % 20 == 0:
            diff = get_material_score(board, ai_color)
            print(f"   Move {moves} | Material Diff: {diff}")

    res = board.result()
    print(f"🏁 Result: {res}")

    # Calculate winner
    if res == "1-0":
        return 1.0 if ai_color == chess.WHITE else 0.0
    elif res == "0-1":
        return 1.0 if ai_color == chess.BLACK else 0.0
    else:
        return 0.5


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("Model not found")
        exit()

    print(f"🤖 Loading AlphaZero Model...")
    model = AlphaZeroModel().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    score_w = play_game(model, chess.WHITE)
    score_b = play_game(model, chess.BLACK)

    print("\n==================================")
    print(f"FINAL SCORE vs Material Bot: {score_w + score_b}/2.0")
    print("==================================")

    if (score_w + score_b) >= 1.5:
        print("✅ SUCCESS! Estimated ELO: ~400")
    elif (score_w + score_b) >= 1.0:
        print("⚠️ DECENT! Estimated ELO: ~300")
    else:
        print("❌ STILL LEARNING. Estimated ELO: < 300")