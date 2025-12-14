import chess
import torch
import random

from models.AlphaZero import AlphaZeroModel
from utils.mcts_move_selector import mcts_select_move


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#MODEL_PATH = "/home/presi/projects/chess_ai_project/training/rl/checkpoints/alpha_zero_rl_main.pth"
MODEL_PATH = "/home/presi/projects/chess_ai_project/training/rl/checkpoints/alpha_zero_rl_checkpoint_ep130.pth"



# ------------------------------------------------------
# EASY BOT 1: RANDOM MOVE BOT
# ------------------------------------------------------
def random_bot(board: chess.Board):
    """Връща напълно случаен легален ход."""
    return random.choice(list(board.legal_moves))


# ------------------------------------------------------
# EASY BOT 2: MATERIAL BOT (избира най-добрата размяна)
# ------------------------------------------------------
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def material_bot(board: chess.Board):
    """Избира хода, който води до най-добра моментална материална печалба."""
    best_move = None
    best_score = -999

    for move in board.legal_moves:
        score = 0
        if board.is_capture(move):
            piece = board.piece_at(move.to_square)
            if piece:
                score += PIECE_VALUES.get(piece.piece_type, 0)

        if score > best_score:
            best_score = score
            best_move = move

    # ако няма печелившо вземане → просто избери случаен ход
    if best_move is None:
        best_move = random.choice(list(board.legal_moves))

    return best_move


# ------------------------------------------------------
# ИГРА НА МАЧОВЕ
# ------------------------------------------------------
def play_match(model, opponent_func, games=20, simulations=100):
    score = 0.0

    for g in range(1, games + 1):
        board = chess.Board()
        is_white = (g % 2 == 1)

        while not board.is_game_over():

            if board.turn == chess.WHITE:
                if is_white:
                    # AlphaZero plays white
                    move = mcts_select_move(model, board, DEVICE, simulations)
                else:
                    move = opponent_func(board)
            else:
                if not is_white:
                    # AlphaZero plays black
                    move = mcts_select_move(model, board, DEVICE, simulations)
                else:
                    move = opponent_func(board)

            board.push(move)

        result = board.result()

        if (result == "1-0" and is_white) or (result == "0-1" and not is_white):
            score += 1
        elif result == "1/2-1/2":
            score += 0.5

        print(f"Game {g}/{games}: result={result}")

    return score / games


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main():
    print("==============================")
    print(" AlphaZero vs EASY BOTS")
    print("==============================")

    print("[INFO] Loading model...")
    model = AlphaZeroModel()
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # -------------------------------
    # Test vs Random Bot
    # -------------------------------
    print("\n=== Testing vs RANDOM BOT ===")
    score_random = play_match(model, random_bot, games=20, simulations=80)
    print(f"→ Avg score vs random bot = {score_random:.3f}")

    # -------------------------------
    # Test vs Material Bot
    # -------------------------------
    print("\n=== Testing vs MATERIAL BOT ===")
    score_material = play_match(model, material_bot, games=20, simulations=80)
    print(f"→ Avg score vs material bot = {score_material:.3f}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
