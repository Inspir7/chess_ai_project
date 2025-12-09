# data/generate_labeled_data.py

import sqlite3
import numpy as np
import chess
import chess.pgn
import random
import multiprocessing
import time

from training.move_encoding import move_to_index, get_total_move_count

TOTAL_MOVES = get_total_move_count()

# ============================================================
# GAME PHASE (same as before)
# ============================================================

def determine_phase(board):
    piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}
    total_value = sum(piece_values[piece.symbol().lower()] for piece in board.piece_map().values())
    if total_value > 20:
        return 0  # opening
    elif total_value > 10:
        return 1  # middlegame
    else:
        return 2  # endgame

# ============================================================
# VALUE LABEL (same as before)
# ============================================================

def evaluate_position(board, result_str):
    if result_str == "1-0":
        return 1.0 if board.turn == chess.WHITE else -1.0
    elif result_str == "0-1":
        return -1.0 if board.turn == chess.WHITE else 1.0
    else:
        return 0.0

# ============================================================
# FEN → (8x8x15)
# ============================================================

def fen_to_tensor(fen):
    board = chess.Board(fen)
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    tensor = np.zeros((8, 8, 15), dtype=np.float32)

    # pieces
    for square, piece in board.piece_map().items():
        x, y = divmod(square, 8)
        tensor[7 - x, y, piece_map[piece.symbol()]] = 1.0

    # side to move
    tensor[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0

    # normalized move number
    tensor[:, :, 13] = board.fullmove_number / 100.0

    # game phase
    tensor[:, :, 14] = determine_phase(board)

    return tensor

# ============================================================
# POLICY LABEL (NOW USING NEW move_encoding.py)
# ============================================================

def build_policy_vector(board, result_str):
    """
    Builds a policy vector (4672,) using the *exact same* move_to_index
    function that RL + MCTS use.
    """
    legal_moves = list(board.legal_moves)
    policy = np.zeros(TOTAL_MOVES, dtype=np.float32)

    if not legal_moves:
        return policy

    # For supervised we use ONE random legal move (same as before),
    # but encoded with the NEW MOVE ENCODING.
    move = random.choice(legal_moves)
    idx = move_to_index(move)

    if 0 <= idx < TOTAL_MOVES:
        policy[idx] = 1.0

    return policy

# ============================================================
# FULL SAMPLE BUILDER
# ============================================================

def fen_to_tensor_with_labels(row):
    fen, result = row
    board = chess.Board(fen)

    state = fen_to_tensor(fen)                    # (8,8,15)
    policy = build_policy_vector(board, result)   # (4672)
    value = evaluate_position(board, result)      # float

    return state, policy, value

# ============================================================
# PARALLEL PROCESSING PIPELINE
# ============================================================

def parallel_process_labeled(fen_rows, filename, num_workers=8, batch_size=10000):
    num_batches = len(fen_rows) // batch_size + (1 if len(fen_rows) % batch_size else 0)

    print(f"Processing {filename} with {num_workers} workers...")
    start_time = time.time()

    with multiprocessing.Pool(num_workers) as pool:
        for i in range(num_batches):
            batch = fen_rows[i * batch_size:(i + 1) * batch_size]
            print(f"Batch {i + 1}/{num_batches} ({len(batch)} samples)...")

            results = pool.map(fen_to_tensor_with_labels, batch)

            states, policies, values = zip(*results)

            np.save(f"{filename}_states_{i}.npy",  np.array(states, dtype=np.float32))
            np.save(f"{filename}_policies_{i}.npy", np.array(policies, dtype=np.float32))
            np.save(f"{filename}_values_{i}.npy",   np.array(values, dtype=np.float32))

            print(f"Saved → {filename}_states_{i}.npy etc.")

    print(f"Finished {filename} in {time.time() - start_time:.2f} seconds.")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()

    print("Loading FEN positions from database...")
    conn = sqlite3.connect("/home/presi/projects/chess_ai_project/data/datachess_games.db")
    cursor = conn.cursor()
    cursor.execute("SELECT fen, result FROM games")
    fen_rows = cursor.fetchall()
    conn.close()

    print(f"Total positions: {len(fen_rows)}")
    random.shuffle(fen_rows)

    train_size = int(0.8 * len(fen_rows))
    val_size   = int(0.1 * len(fen_rows))

    train_rows = fen_rows[:train_size]
    val_rows   = fen_rows[train_size:train_size + val_size]
    test_rows  = fen_rows[train_size + val_size:]

    print(f"Split: Train={len(train_rows)}, Val={len(val_rows)}, Test={len(test_rows)}")

    parallel_process_labeled(train_rows, "train_labeled")
    parallel_process_labeled(val_rows,   "val_labeled",  batch_size=5000)
    parallel_process_labeled(test_rows,  "test_labeled", batch_size=5000)

    print("All labeled .npy files saved successfully!")