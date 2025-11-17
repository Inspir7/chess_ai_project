import chess

# 8x8 board = 64 squares
# 73 possible moves per square:
# 56 for knight, queen, rook, bishop, king, promotions
# + 17 underpromotions, special cases etc.
MAX_MOVES = 4672

def move_to_index(move, board):
    """
    Encode a chess.Move into an index in the 4672-d policy output vector.
    This assumes the AlphaZero-style encoding.
    """
    from_square = move.from_square
    to_square = move.to_square
    move_diff = to_square - from_square

    row_from = from_square // 8
    col_from = from_square % 8
    row_to = to_square // 8
    col_to = to_square % 8
    d_row = row_to - row_from
    d_col = col_to - col_from

    # Promotion moves
    if move.promotion:
        promo_offset = {'q': 0, 'r': 1, 'b': 2, 'n': 3}[chess.piece_symbol(move.promotion).lower()]
        dir_offset = 56 + promo_offset
        return from_square * 73 + dir_offset

    # Basic directional moves (56 directions encoded)
    # Simplified directional encoding idea
    direction_index = direction_to_index(d_row, d_col)
    if direction_index is None:
        return None
    return from_square * 73 + direction_index

def index_to_move(index, board):
    """
    Decode index from 4672-d vector into a legal chess.Move for this board.
    """
    from_square = index // 73
    offset = index % 73
    legal_moves = list(board.legal_moves)

    if offset >= 56:
        promo_piece = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT][offset - 56]
        for move in legal_moves:
            if move.from_square == from_square and move.promotion == promo_piece:
                return move
    else:
        d_row, d_col = index_to_direction(offset)
        row = from_square // 8
        col = from_square % 8
        to_row = row + d_row
        to_col = col + d_col
        if 0 <= to_row < 8 and 0 <= to_col < 8:
            to_square = to_row * 8 + to_col
            candidate = chess.Move(from_square, to_square)
            if candidate in legal_moves:
                return candidate
    return None

def direction_to_index(d_row, d_col):
    """
    Map a (d_row, d_col) to a unique direction index from 0 to 55 + 8 (64 total)
    """
    directions = []

    # Линейни и диагонални
    for dr in range(-7, 8):
        for dc in range(-7, 8):
            if dr == 0 and dc == 0:
                continue
            if abs(dr) == abs(dc) or dr == 0 or dc == 0:
                directions.append((dr, dc))

    # Добавяме конски ходове (L-образни)
    knight_moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    directions.extend(knight_moves)

    try:
        return directions.index((d_row, d_col))
    except ValueError:
        return None


def index_to_direction(index):
    """
    Reverse of direction_to_index: maps index [0, 55] to (d_row, d_col).
    """
    directions = []
    for dr in range(-7, 8):
        for dc in range(-7, 8):
            if dr == 0 and dc == 0:
                continue
            if abs(dr) == abs(dc) or dr == 0 or dc == 0:
                directions.append((dr, dc))
    return directions[index]

def get_total_move_count():
    """
    Връща общия брой възможни ходове в AlphaZero-style encoding.
    """
    return MAX_MOVES