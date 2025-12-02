import chess
import numpy as np
import random

# ==========================
#  Цветове и размери (GUI)
# ==========================
LIGHT_SQUARE = (255, 228, 225)
DARK_SQUARE = (255, 182, 193)
HIGHLIGHT = (255, 240, 245)
LAST_MOVE_SRC = (255, 182, 193)
LAST_MOVE_DST = (255, 105, 180)
SQUARE_SIZE = 75


# ==========================
#  Базови функции за дъската
# ==========================
def initial_board():
    """Създава нова стандартна шахматна дъска."""
    return chess.Board()


def get_square_from_mouse(pos):
    """
    Конвертира пикселни координати (x, y) към chess.square,
    при условие, че квадратите са SQUARE_SIZE x SQUARE_SIZE.
    Използва се само от GUI.
    """
    col = pos[0] // SQUARE_SIZE
    row = pos[1] // SQUARE_SIZE
    square = chess.square(col, 7 - row)
    return square


def make_move(board, from_square, to_square, promotion=None):
    """
    Прави ход върху board от from_square до to_square.

    - Ако move е валиден като елементарен ход или с подадена промоция → push.
    - Ако не, пробва всички възможни промоции (Q, R, B, N).
    Връща True при успешен ход, иначе False.
    """
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


# ==========================
#  GUI: рисуване на дъската
# ==========================
def draw_board(screen, board, selected_square=None, last_move=None):
    """
    Рисува дъската и фигурите върху подадения pygame surface `screen`.

    ВАЖНО:
    - Прави локален import на pygame вътре в функцията.
    - Така training/multiprocessing кодът може да импортва chess_utils
      без да дърпа pygame и без да блокира worker-и.
    """
    import pygame  # lazy import, безопасен за RL worker-и

    font = pygame.font.SysFont("dejavusans", 48)

    # Квадратите
    for row in range(8):
        for col in range(8):
            color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE

            # Оцветяване на последния ход
            if last_move:
                src, dst = last_move
                src_col, src_row = chess.square_file(src), 7 - chess.square_rank(src)
                dst_col, dst_row = chess.square_file(dst), 7 - chess.square_rank(dst)
                if row == src_row and col == src_col:
                    color = LAST_MOVE_SRC
                elif row == dst_row and col == dst_col:
                    color = LAST_MOVE_DST

            # Оцветяване на селектиран квадрат
            if selected_square is not None:
                sq_col = chess.square_file(selected_square)
                sq_row = 7 - chess.square_rank(selected_square)
                if row == sq_row and col == sq_col:
                    color = HIGHLIGHT

            rect = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, color, rect)

    # Фигурите
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            text = font.render(piece.unicode_symbol(), True, (0, 0, 0))
            screen.blit(text, (col * SQUARE_SIZE + 15, row * SQUARE_SIZE + 10))


def choose_promotion(board, from_square, to_square, player="human", policy=None):
    """
    Логика за избор на промоция.

    - Ако player=="human" → няма автоматична промоция (GUI трябва да пита).
    - Ако player=="ai" и има policy dict (move -> prob):
        избира промоция с най-голяма вероятност.
    - В останалите случаи → промоция на дама.
    """
    if player == "human":
        # GUI слой трябва да пита потребителя
        return None
    elif player == "ai" and policy:
        best_prob = -1.0
        best_piece = chess.QUEEN
        for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            test_move = chess.Move(from_square, to_square, promotion=piece_type)
            if test_move in policy and policy[test_move] > best_prob:
                best_prob = policy[test_move]
                best_piece = piece_type
        return chess.Move(from_square, to_square, promotion=best_piece)
    else:
        return chess.Move(from_square, to_square, promotion=chess.QUEEN)


# ==========================
#  Тензорни представяния
# ==========================
def board_to_tensor(board):
    """
    Конвертира текущата шахматна дъска в 8x8x15 numpy тензор.

    Канали:
    - 0-5: бели фигури (P, N, B, R, Q, K)
    - 6-11: черни фигури (p, n, b, r, q, k)
    - 12: чий е ходът (1 за бели, 0 за черни)
    - 13: флаг (примерен) за рокада на белите (1 ако имат kingside castling право)
    - 14: флаг (примерен) за рокада на черните (1 ако имат kingside castling право)

    Забележка:
    - Форматът е (8, 8, 15) — HWC.
    - Моделът ти очаква (15, 8, 8), т.е. трябва да се транспонира:
      tensor.transpose(2, 0, 1)
      или в PyTorch: x = x.permute(0, 3, 1, 2)
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

    # Рокади (тук ползваме kingside като примерен флаг)
    tensor[:, :, 13] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[:, :, 14] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0

    return tensor


# ==========================
#  Достъпни ходове / край на играта
# ==========================
def get_legal_moves(board):
    """Връща списък от всички легални ходове на дадено board."""
    return list(board.legal_moves)


def get_game_result(board):
    """
    Връща резултата от гледна точка на играча, който току-що е играл (т.е. board.turn е след хода):

    - 1  → победа за току-що игралата страна
    - -1 → загуба
    - 0  → реми
    - None → играта не е приключила
    """
    if board.is_checkmate():
        # ако е мат и е ход на черните → белите са матували, бял печели (1)
        # ако е мат и е ход на белите → черните са матували, черен печели (-1)
        return 1 if board.turn == chess.BLACK else -1

    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_seventyfive_moves()
        or board.is_fivefold_repetition()
    ):
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

    - 1  → победа
    - -1 → загуба
    - 0  → реми
    - None → играта не е приключила

    player_color: chess.WHITE или chess.BLACK
    """
    if board.is_checkmate():
        # board.turn е страната, която е на ход след мата (загубилата страна)
        return 1 if board.turn != player_color else -1
    elif (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_seventyfive_moves()
        or board.is_fivefold_repetition()
    ):
        return 0
    else:
        return None