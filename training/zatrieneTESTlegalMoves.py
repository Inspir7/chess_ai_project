# test_move_encoding_random_positions.py
import chess
import random
from models.move_encoding import move_to_index

def generate_random_positions(num_positions=100):
    """Generate random legal positions by playing random moves from the starting board."""
    positions = []
    board = chess.Board()
    for _ in range(num_positions):
        board_copy = board.copy()
        positions.append(board_copy)
        # Play 1–5 random moves to reach new position
        for _ in range(random.randint(1, 5)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(random.choice(moves))
    return positions

def test_all_positions(positions):
    errors = 0
    for idx, board in enumerate(positions):
        for move in board.legal_moves:
            move_idx = move_to_index(move)
            if move_idx == -1:
                print(f"[ERROR] Position {idx}: move_to_index returned -1 for move {move} on FEN: {board.fen()}")
                errors += 1
    if errors == 0:
        print("[PASS] All legal moves in all positions have valid indices!")
    else:
        print(f"[FAIL] {errors} moves missing in move encoding across {len(positions)} positions.")

if __name__ == "__main__":
    print("[INFO] Generating random positions...")
    positions = generate_random_positions(num_positions=100)
    print("[INFO] Testing move encoding on these positions...")
    test_all_positions(positions)
