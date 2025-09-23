import sqlite3
import numpy as np
import chess
import chess.pgn
import matplotlib.pyplot as plt
import seaborn as sns


# Функция за определяне на фазата на играта
def determine_phase(board):
    piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}
    total_value = sum(piece_values[piece.symbol().lower()] for piece in board.piece_map().values())
    print(f"Total piece value: {total_value}")
    if total_value > 20:
        print("Assigned phase: 0 (Opening)")
        return 0  # Дебют
    elif total_value > 10:
        print("Assigned phase: 1 (Middlegame)")
        return 1  # Миттелшпил
    else:
        print("Assigned phase: 2 (Endgame)")
        return 2  # Ендшпил


# Функция за конвертиране на FEN в 8x8x15 тензор (добавени логове)
def fen_to_tensor(fen):
    print(f"\nProcessing FEN: {fen}")
    board = chess.Board(fen)
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # Бели фигури
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Черни фигури
    }
    tensor = np.zeros((8, 8, 15), dtype=np.int8)

    for square, piece in board.piece_map().items():
        x, y = divmod(square, 8)
        tensor[7 - x, y, piece_map[piece.symbol()]] = 1  # Координатна трансформация

    print(f"Board turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    tensor[:, :, 12] = 1 if board.turn == chess.WHITE else 0  # Чий ред е

    print(f"Full move number (before normalization): {board.fullmove_number}")
    tensor[:, :, 13] = board.fullmove_number / 100  # Номер на хода (нормализиран)
    print(f"Move number tensor value (normalized): {tensor[0, 0, 13]}")

    tensor[:, :, 14] = determine_phase(board)  # Фаза на играта
    print(f"Game phase (tensor value): {tensor[0, 0, 14]}")

    print(f"Sample tensor data (first row, first column): {tensor[0, 0, :]}")
    return tensor


# Свързване с базата и извличане на 5 FEN позиции
conn = sqlite3.connect("datachess_games.db")
cursor = conn.cursor()
cursor.execute("SELECT fen FROM games LIMIT 5")
fen_positions = [row[0] for row in cursor.fetchall()]
conn.close()

# Преобразуване на FEN позиции в тензори
tensors = [fen_to_tensor(fen) for fen in fen_positions]

# Визуализиране на няколко канала
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
channels = [0, 3, 6, 9, 12, 14]  # Пешки, топове, чейнъл за ред на играча, фаза на играта
channel_names = ["White Pawns", "White Rooks", "Black Pawns", "Black Rooks", "Turn", "Game Phase"]

for i, ax in enumerate(axes.flat):
    sns.heatmap(tensors[0][:, :, channels[i]], cmap="coolwarm", annot=True, cbar=False, ax=ax)
    ax.set_title(channel_names[i])
    ax.axis("off")

plt.tight_layout()
plt.show()
