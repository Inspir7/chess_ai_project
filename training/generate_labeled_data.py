import sqlite3
import numpy as np
import chess
import chess.pgn
import random
import multiprocessing
import time

# ======================
# Игра фаза (за channel 14)
# ======================
def determine_phase(board):
    piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}
    total_value = sum(piece_values[piece.symbol().lower()] for piece in board.piece_map().values())
    if total_value > 20:
        return 0  # Дебют
    elif total_value > 10:
        return 1  # Миттелшпил
    else:
        return 2  # Ендшпил

# ======================
# Индексиране на ходовете (policy)
# ======================
def move_to_index(move):
    from_square = move.from_square
    to_square = move.to_square
    promotion = move.promotion

    index = 64 * from_square + to_square

    if promotion is not None:
        promotion_offset = {
            chess.KNIGHT: 0,
            chess.BISHOP: 1,
            chess.ROOK: 2,
            chess.QUEEN: 3
        }[promotion]
        index += 4096 + promotion_offset * 64

    return index

def move_to_one_hot(move):
    index = move_to_index(move)
    vec = np.zeros(4672, dtype=np.float32)
    if 0 <= index < 4672:
        vec[index] = 1.0
    return vec

# ======================
# Оценка на позицията на база резултат
# ======================
def evaluate_position(board, result_str):
    # Резултат: '1-0', '0-1', '1/2-1/2'
    if result_str == '1-0':
        return 1.0 if board.turn == chess.WHITE else -1.0
    elif result_str == '0-1':
        return -1.0 if board.turn == chess.WHITE else 1.0
    else:
        return 0.0

# ======================
# FEN към 8x8x15 + labels
# ======================
def fen_to_tensor(fen):
    board = chess.Board(fen)
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    tensor = np.zeros((8, 8, 15), dtype=np.int8)

    for square, piece in board.piece_map().items():
        x, y = divmod(square, 8)
        tensor[7 - x, y, piece_map[piece.symbol()]] = 1

    tensor[:, :, 12] = 1 if board.turn == chess.WHITE else 0
    tensor[:, :, 13] = board.fullmove_number / 100
    tensor[:, :, 14] = determine_phase(board)

    return tensor

def fen_to_tensor_with_labels(row):
    fen, result = row
    board = chess.Board(fen)
    tensor = fen_to_tensor(fen)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        policy = np.zeros(4672, dtype=np.float32)
    else:
        move = random.choice(legal_moves)
        policy = move_to_one_hot(move)

    value = evaluate_position(board, result)

    return tensor, policy, value

# ======================
# Паралелна обработка на данни
# ======================
def parallel_process_labeled(fen_rows, filename, num_workers=8, batch_size=10000):
    num_batches = len(fen_rows) // batch_size + (1 if len(fen_rows) % batch_size else 0)

    print(f"Обработка на {filename} с {num_workers} работни процеса... (с policy и value)")
    start_time = time.time()

    with multiprocessing.Pool(num_workers) as pool:
        for i in range(num_batches):
            batch = fen_rows[i * batch_size:(i + 1) * batch_size]
            print(f"Обработка на партида {i + 1}/{num_batches} ({len(batch)} примера)...")
            results = pool.map(fen_to_tensor_with_labels, batch)

            tensors, policies, values = zip(*results)
            np.save(f"{filename}_states_{i}.npy", np.array(tensors, dtype=np.int8))
            np.save(f"{filename}_policies_{i}.npy", np.array(policies, dtype=np.float32))
            np.save(f"{filename}_values_{i}.npy", np.array(values, dtype=np.float32))

            print(f"Записана партида {i + 1}/{num_batches} в {filename}_states/ policies/ values")

    elapsed_time = time.time() - start_time
    print(f"Обработка на {filename} завършена за {elapsed_time:.2f} секунди!")

# ======================
# Основна програма
# ======================
if __name__ == '__main__':
    multiprocessing.freeze_support()

    print("Извличане на FEN позиции и резултати от базата данни...")
    conn = sqlite3.connect("C:\\Users\\prezi\\PycharmProjects\\chess_ai_project\\data\\datachess_games.db")
    cursor = conn.cursor()
    cursor.execute("SELECT fen, result FROM games")
    fen_rows = cursor.fetchall()
    conn.close()
    print(f"Общо позиции: {len(fen_rows)}")

    print("Разбъркване на позициите...")
    random.shuffle(fen_rows)

    train_size = int(0.8 * len(fen_rows))
    val_size = int(0.1 * len(fen_rows))

    train_rows = fen_rows[:train_size]
    val_rows = fen_rows[train_size:train_size + val_size]
    test_rows = fen_rows[train_size + val_size:]

    print(f"Разделение: Train: {len(train_rows)}, Validation: {len(val_rows)}, Test: {len(test_rows)}")

    parallel_process_labeled(train_rows, "train_labeled")
    parallel_process_labeled(val_rows, "val_labeled", batch_size=5000)
    parallel_process_labeled(test_rows, "test_labeled", batch_size=5000)

    print("Всички .npy файлове са записани успешно!")
