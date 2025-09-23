import sqlite3
import numpy as np
import chess
import chess.pgn
import random
import multiprocessing
import time


# Функция за определяне на фазата на играта
def determine_phase(board):
    piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}
    total_value = sum(piece_values[piece.symbol().lower()] for piece in board.piece_map().values())
    if total_value > 20:
        return 0  # Дебют
    elif total_value > 10:
        return 1  # Миттелшпил
    else:
        return 2  # Ендшпил


# Функция за конвертиране на FEN в 8x8x15 тензор
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


# Функция за обработка на партиди
def process_fen_batch(fen_batch):
    return [fen_to_tensor(fen) for fen in fen_batch]


# Паралелна обработка на FEN позиции
def parallel_process(fen_list, filename, num_workers=8, batch_size=10000):
    num_batches = len(fen_list) // batch_size + (1 if len(fen_list) % batch_size else 0)

    print(f"Обработка на {filename} с {num_workers} работни процеса...")
    start_time = time.time()

    with multiprocessing.Pool(num_workers) as pool:
        for i in range(num_batches):
            batch = fen_list[i * batch_size:(i + 1) * batch_size]
            print(f"Обработка на партида {i + 1}/{num_batches} ({len(batch)} примера)...")
            results = pool.map(fen_to_tensor, batch)
            np.save(f"{filename}_{i}.npy", np.array(results))
            print(f"Записана партида {i + 1}/{num_batches} в {filename}_{i}.npy")

    elapsed_time = time.time() - start_time
    print(f"Обработка на {filename} завършена за {elapsed_time:.2f} секунди!")


# Основна програма
if __name__ == '__main__':
    multiprocessing.freeze_support()

    print("Извличане на FEN позиции от базата данни...")
    conn = sqlite3.connect("datachess_games.db")
    cursor = conn.cursor()
    cursor.execute("SELECT fen FROM games")
    fen_positions = [row[0] for row in cursor.fetchall()]
    conn.close()
    print(f"Общо FEN позиции: {len(fen_positions)}")

    # Разбъркване на данните
    print("Разбъркване на FEN позициите...")
    random.shuffle(fen_positions)

    # Разделяне на Train (80%), Validation (10%), Test (10%)
    train_size = int(0.8 * len(fen_positions))
    val_size = int(0.1 * len(fen_positions))

    train_fen = fen_positions[:train_size]
    val_fen = fen_positions[train_size:train_size + val_size]
    test_fen = fen_positions[train_size + val_size:]

    print(f"Разделение на данни: Train: {len(train_fen)}, Validation: {len(val_fen)}, Test: {len(test_fen)}")

    # Обработка и запис
    parallel_process(train_fen, "train_tensors")
    parallel_process(val_fen, "val_tensors", batch_size=5000)
    parallel_process(test_fen, "test_tensors", batch_size=5000)

    print("Всички данни са записани успешно в .npy файлове!")
