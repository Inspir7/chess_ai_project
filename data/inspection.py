import sqlite3
import numpy as np
import chess
import chess.pgn
import random
import multiprocessing
import os


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


# Функция за инспекция на записаните данни
def inspect_data(folder):
    print(f"\nИнспектиране на данните в {folder}...")
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    if not files:
        print(f"Няма намерени .npy файлове в {folder}")
        return

    for file in files[:3]:  # Инспектираме до 3 файла за преглед
        file_path = os.path.join(folder, file)
        print(f"\nЗареждане на файл: {file_path}")
        data = np.load(file_path, mmap_mode='r')
        print(f"Файл: {file_path}")
        print(f"Форма на тензорите: {data.shape}")
        print(f"Брой записани позиции: {len(data)}")
        print(f"Примерен тензор: {data[0]}")
        print("-" * 50)
    print(f"Инспекцията на {folder} завърши.\n")


# Папки с данни
data_dir = "C:\\Users\\prezi\\PycharmProjects\\chess_ai_project\\data\\datasets"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "validation")
test_dir = os.path.join(data_dir, "test")

# Инспекция на първите файлове от train, validation и test
inspect_data(train_dir)
inspect_data(val_dir)
inspect_data(test_dir)

print("\nПроцесът на инспекция приключи успешно!")
