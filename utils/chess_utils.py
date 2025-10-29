import pygame
import chess
import numpy as np
import random

# --- Цветове и размери ---
LIGHT_SQUARE = (255, 228, 225)
DARK_SQUARE = (255, 182, 193)
HIGHLIGHT = (255, 240, 245)
LAST_MOVE_SRC = (255, 182, 193)
LAST_MOVE_DST = (255, 105, 180)
SQUARE_SIZE = 75

# --- Инициализация на дъската ---
def initial_board():
    return chess.Board()

# --- Преобразуване от пиксели към square ---
def get_square_from_mouse(pos):
    col = pos[0] // SQUARE_SIZE
    row = pos[1] // SQUARE_SIZE
    square = chess.square(col, 7 - row)
    return square

# --- Извършване на ход с валидност и промоция ---
def make_move(board, from_square, to_square, promotion=None):
    try:
        move = chess.Move(from_square, to_square, promotion=promotion)
        if move in board.legal_moves:
            board.push(move)
            return True
        else:
            for promo_piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                promo_move = chess.Move(from_square, to_square, promotion=promo_piece)
                if promo_move in board.legal_moves:
                    board.push(promo_move)
                    return True
    except Exception as e:
        print(f"Invalid move: {e}")
    return False

# --- Рисуване на дъската и фигурите ---
def draw_board(screen, board, selected_square=None, last_move=None):
    font = pygame.font.SysFont("dejavusans", 48)
    for row in range(8):
        for col in range(8):
            color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE

            if last_move:
                src, dst = last_move
                src_col, src_row = chess.square_file(src), 7 - chess.square_rank(src)
                dst_col, dst_row = chess.square_file(dst), 7 - chess.square_rank(dst)
                if row == src_row and col == src_col:
                    color = LAST_MOVE_SRC
                elif row == dst_row and col == dst_col:
                    color = LAST_MOVE_DST

            if selected_square is not None:
                sq_col = chess.square_file(selected_square)
                sq_row = 7 - chess.square_rank(selected_square)
                if row == sq_row and col == sq_col:
                    color = HIGHLIGHT

            rect = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, color, rect)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            text = font.render(piece.unicode_symbol(), True, (0, 0, 0))
            screen.blit(text, (col * SQUARE_SIZE + 15, row * SQUARE_SIZE + 10))

# --- Избор на промоция ---
def choose_promotion(board, from_square, to_square, player="human", policy=None):
    if player == "human":
        return None
    elif player == "ai" and policy:
        best_prob = -1
        best_piece = chess.QUEEN
        for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            test_move = chess.Move(from_square, to_square, promotion=piece_type)
            if test_move in policy and policy[test_move] > best_prob:
                best_prob = policy[test_move]
                best_piece = piece_type
        return chess.Move(from_square, to_square, promotion=best_piece)
    else:
        return chess.Move(from_square, to_square, promotion=chess.QUEEN)

# ===========================
# --- НОВИ FUNKЦИИ ---
# ===========================

def board_to_tensor(board):
    """
    Конвертира текущата шахматна дъска в 8x8x15 numpy тензор.
    Канали:
    - 0-5: бели фигури (P, N, B, R, Q, K)
    - 6-11: черни фигури (p, n, b, r, q, k)
    - 12: чий е хода (1 за бели, 0 за черни)
    - 13: флаг за рокада (бяла)
    - 14: флаг за рокада (черна)
    """
    tensor = np.zeros((8, 8, 15), dtype=np.float32)
    piece_map = board.piece_map()

    piece_to_index = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }

    for square, piece in piece_map.items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        tensor[row, col, piece_to_index[piece.symbol()]] = 1.0

    # Ход на играча
    tensor[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0

    # Рокади
    tensor[:, :, 13] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[:, :, 14] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0

    return tensor

def get_legal_moves(board):
    return list(board.legal_moves)

def move_to_index(move):
    """ Конвертира ход в индекс за policy вектор (0-4671) """
    return move.from_square * 64 + move.to_square

def index_to_move(index):
    """ Възстановява шахматен ход от policy индекс """
    from_square = index // 64
    to_square = index % 64
    return chess.Move(from_square, to_square)

def get_game_result(board):
    if board.is_checkmate():
        return 1 if board.turn == chess.BLACK else -1
    if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0
    return None


def game_over(board):
    """
    Проверява дали играта е приключила (мат, пат, реми и т.н.)
    """
    return board.is_game_over()


def result_from_perspective(board, player_color):
    """
    Връща резултата от гледна точка на дадения играч:
    - 1 → победа
    - -1 → загуба
    - 0 → реми
    """
    if board.is_checkmate():
        return 1 if board.turn != player_color else -1
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0
    else:
        return None
