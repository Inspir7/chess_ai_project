import sqlite3
import numpy as np
import chess
import random
import multiprocessing
import time
import os

from training.move_encoding import move_to_index, get_total_move_count

TOTAL_MOVES = get_total_move_count()


# ============================================================
# 1. HELPER FUNCTIONS
# ============================================================

def determine_phase(board):
    """Определя фазата на играта (0=Opening, 1=Mid, 2=End)"""
    piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}
    total_value = sum(piece_values[piece.symbol().lower()] for piece in board.piece_map().values())
    if total_value > 20:
        return 0
    elif total_value > 10:
        return 1
    else:
        return 2


def get_canonical_data(board, next_move):
    """
    Помощна функция за Supervised Learning.
    Връща завъртяна дъска и завъртян ход.
    """
    if board.turn == chess.WHITE:
        return board, next_move
    else:
        # Завъртаме дъската
        board_flipped = board.transform(chess.flip_vertical)
        board_flipped = board_flipped.transform(chess.flip_horizontal)

        # Завъртаме и хода
        if next_move:
            def flip_square(sq):
                return chess.square(7 - chess.square_file(sq), 7 - chess.square_rank(sq))

            from_sq = flip_square(next_move.from_square)
            to_sq = flip_square(next_move.to_square)
            move_flipped = chess.Move(from_sq, to_sq, promotion=next_move.promotion)
            return board_flipped, move_flipped

        return board_flipped, None


def fen_to_tensor(input_data):
    """
    Приема chess.Board или FEN.
    Връща (8,8,15) тензор.
    АВТОМАТИЧНО ЗАВЪРТА ДЪСКАТА (Fix за RL "Suicide" bug).
    """
    # 1. Input Check
    if isinstance(input_data, str):
        board = chess.Board(input_data)
    else:
        board = input_data

    # 2. PERSPECTIVE FLIP (Липсваше в твоя код!)
    # Ако е ред на черните, обръщаме дъската, за да изглежда като за бели
    if board.turn == chess.BLACK:
        board = board.transform(chess.flip_vertical)
        board = board.transform(chess.flip_horizontal)

    # Mapping
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    tensor = np.zeros((8, 8, 15), dtype=np.float32)

    for square, piece in board.piece_map().items():
        x, y = divmod(square, 8)
        tensor[7 - x, y, piece_map[piece.symbol()]] = 1.0

    tensor[:, :, 12] = 1.0
    tensor[:, :, 13] = board.fullmove_number / 100.0
    tensor[:, :, 14] = determine_phase(board)

    return tensor


def get_value_target(result_str, original_turn):
    if result_str == "1-0":
        winner = chess.WHITE
    elif result_str == "0-1":
        winner = chess.BLACK
    else:
        return 0.0

    if winner == original_turn:
        return 1.0
    else:
        return -1.0


# ============================================================
# 2. GAME PROCESSOR (WORKER)
# ============================================================

def process_game(row):
    """Обработва ЕДНА игра."""
    moves_uci_str, result_str = row
    if not moves_uci_str: return []

    board = chess.Board()
    samples = []

    try:
        moves = moves_uci_str.split()
        for move_str in moves:
            next_move = chess.Move.from_uci(move_str)

            original_turn = board.turn

            # ВАЖНО: Тук променяме логиката, за да не завъртаме два пъти!

            # А) За POLICY (Target) ни трябва завъртяният ход:
            _, canon_move = get_canonical_data(board, next_move)

            # Б) За TENSOR (Input) подаваме СУРОВАТА дъска на fen_to_tensor
            # (Тя вече сама ще си я завърти вътре)
            state_tensor = fen_to_tensor(board)

            # Create Policy Vector
            policy_vector = np.zeros(TOTAL_MOVES, dtype=np.float32)
            idx = move_to_index(canon_move)
            if 0 <= idx < TOTAL_MOVES:
                policy_vector[idx] = 1.0

            # Value Target
            value_target = get_value_target(result_str, original_turn)

            samples.append((state_tensor, policy_vector, value_target))
            board.push(next_move)

    except ValueError:
        pass

    return samples


# ============================================================
# 3. PARALLEL MANAGER
# ============================================================

def save_batch(samples, folder_path, prefix, batch_idx):
    if not samples: return
    states, policies, values = zip(*samples)

    path_states = os.path.join(folder_path, f"{prefix}_states_{batch_idx}.npy")
    path_policies = os.path.join(folder_path, f"{prefix}_policies_{batch_idx}.npy")
    path_values = os.path.join(folder_path, f"{prefix}_values_{batch_idx}.npy")

    np.save(path_states, np.array(states, dtype=np.float32))
    np.save(path_policies, np.array(policies, dtype=np.float32))
    np.save(path_values, np.array(values, dtype=np.float32))

    print(f"  -> Saved batch {batch_idx} in '{folder_path}' ({len(samples)} pos)")


def parallel_process_games(rows, output_folder, file_prefix, num_workers=6, games_per_chunk=1000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    print(f"Starting processing for '{file_prefix}' -> {output_folder} ...")

    pool = multiprocessing.Pool(num_workers)
    total_samples_buffer = []
    batch_counter = 0

    for i in range(0, len(rows), games_per_chunk):
        chunk = rows[i: i + games_per_chunk]
        results = pool.map(process_game, chunk)

        for game_samples in results:
            total_samples_buffer.extend(game_samples)

        if len(total_samples_buffer) >= 50000:
            save_batch(total_samples_buffer, output_folder, file_prefix, batch_counter)
            total_samples_buffer = []
            batch_counter += 1

    if total_samples_buffer:
        save_batch(total_samples_buffer, output_folder, file_prefix, batch_counter)

    pool.close()
    pool.join()


# ============================================================
# 4. MAIN
# ============================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()

    PROJECT_ROOT = "/home/presi/projects/chess_ai_project"
    DB_PATH = os.path.join(PROJECT_ROOT, "data/datachess_games.db")

    TRAIN_DIR = os.path.join(PROJECT_ROOT, "data/processed/train")
    VAL_DIR = os.path.join(PROJECT_ROOT, "data/processed/val")
    TEST_DIR = os.path.join(PROJECT_ROOT, "data/processed/test")

    print(f"Reading games from {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT moves_uci, result FROM games WHERE moves_uci IS NOT NULL LIMIT 20000")

    rows = cursor.fetchall()
    conn.close()

    print(f"Loaded {len(rows)} games total.")
    random.shuffle(rows)

    n = len(rows)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_games = rows[:train_end]
    val_games = rows[train_end:val_end]
    test_games = rows[val_end:]

    print(f"Splitting data: Train={len(train_games)}, Val={len(val_games)}, Test={len(test_games)}")

    parallel_process_games(train_games, TRAIN_DIR, "train")
    parallel_process_games(val_games, VAL_DIR, "val")
    parallel_process_games(test_games, TEST_DIR, "test")

    print("\n✅ READY! All data generated correctly.")