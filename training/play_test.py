# training/play_test_match.py
import chess
import chess.pgn
import torch
import time

from models.AlphaZero import AlphaZeroModel
from training.mcts import MCTS
from training.move_encoding import get_total_move_count
from data.generate_labeled_data import fen_to_tensor


def choose_move_from_mcts(board, model, device, sims=100, temperature=0.0):
    """Run MCTS and pick the best move deterministically."""
    mcts = MCTS(model, device, simulations=sims)
    pi_dict = mcts.run(board, move_number=board.fullmove_number)

    # convert dictionary to vector
    moves, probs = zip(*pi_dict.items())
    probs = torch.tensor(probs)

    # deterministic choice (argmax)
    best_id = torch.argmax(probs).item()
    return moves[best_id], pi_dict


def play_single_game(model, device, sims=100, out_pgn_path=None, verbose=False):
    """Plays one model-vs-model game and optionally save PGN."""
    board = chess.Board()
    game = chess.pgn.Game()
    node = game

    move_history = []

    ply = 0
    while not board.is_game_over() and ply < 200:
        move, pi = choose_move_from_mcts(board, model, device, sims=sims)

        # record to PGN
        node = node.add_variation(move)
        board.push(move)
        ply += 1

        if verbose:
            print(f"[{ply}] {move}   FEN: {board.fen()}")

        move_history.append((ply, move, dict(pi)))

    result = board.result()
    game.headers["Result"] = result
    game.headers["Event"] = "TestMatch"
    game.headers["Site"] = "Local"
    game.headers["White"] = "AlphaZero(net)"
    game.headers["Black"] = "AlphaZero(net)"
    game.headers["Date"] = time.strftime("%Y.%m.%d")

    if out_pgn_path:
        with open(out_pgn_path, "w") as f:
            f.write(str(game))

    return result, move_history, str(game)


def main():
    print("=== AlphaZero Test Match ===")

    model_path = "alpha_zero_supervised.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    print(f"[INFO] Loading model: {model_path}")
    model = AlphaZeroModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("[INFO] Running model vs model test game...")
    result, history, pgn = play_single_game(
        model,
        device,
        sims=100,
        out_pgn_path="test_game.pgn",
        verbose=True
    )

    print("\n=== GAME FINISHED ===")
    print(f"Result: {result}")
    print("Saved PGN → test_game.pgn")


if __name__ == "__main__":
    main()