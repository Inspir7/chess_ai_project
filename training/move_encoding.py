import chess

MOVE_TYPES_PER_FROM = 73
TOTAL_MOVES = 64 * MOVE_TYPES_PER_FROM

# Directions (dx, dy): x = file (column), y = rank (row)
DIRECTIONS = [
    (1, 0), (-1, 0), (0, 1), (0, -1),     # rook-like
    (1, 1), (1, -1), (-1, 1), (-1, -1)    # bishop-like
]

KNIGHT_OFFSETS = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1)
]

PROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]

ALL_MOVES = []


def in_bounds(file, rank):
    return 0 <= file < 8 and 0 <= rank < 8


for from_sq in range(64):
    file = chess.square_file(from_sq)
    rank = chess.square_rank(from_sq)
    move_type = 0

    # Sliding moves (8 directions × up to 7 steps)
    for dx, dy in DIRECTIONS:
        for dist in range(1, 8):
            tx, ty = file + dx * dist, rank + dy * dist
            if in_bounds(tx, ty):
                to_sq = chess.square(tx, ty)
                ALL_MOVES.append(chess.Move(from_sq, to_sq))
                move_type += 1
            else:
                break

    # Knight moves (8 offsets)
    for dx, dy in KNIGHT_OFFSETS:
        tx, ty = file + dx, rank + dy
        if in_bounds(tx, ty):
            to_sq = chess.square(tx, ty)
            ALL_MOVES.append(chess.Move(from_sq, to_sq))
            move_type += 1

    # Pawn promotions (both colors)
    if rank == 6:  # white pawn promotions
        for dx in [-1, 0, 1]:
            tx, ty = file + dx, rank + 1
            if in_bounds(tx, ty):
                to_sq = chess.square(tx, ty)
                for promo_piece in PROMO_PIECES:
                    ALL_MOVES.append(chess.Move(from_sq, to_sq, promotion=promo_piece))
                    move_type += 1
    if rank == 1:  # black pawn promotions
        for dx in [-1, 0, 1]:
            tx, ty = file + dx, rank - 1
            if in_bounds(tx, ty):
                to_sq = chess.square(tx, ty)
                for promo_piece in PROMO_PIECES:
                    ALL_MOVES.append(chess.Move(from_sq, to_sq, promotion=promo_piece))
                    move_type += 1

    # Add castling moves for king squares
    if from_sq in [chess.E1, chess.E8]:
        if from_sq == chess.E1:
            ALL_MOVES.append(chess.Move.from_uci("e1g1"))
            ALL_MOVES.append(chess.Move.from_uci("e1c1"))
            move_type += 2
        elif from_sq == chess.E8:
            ALL_MOVES.append(chess.Move.from_uci("e8g8"))
            ALL_MOVES.append(chess.Move.from_uci("e8c8"))
            move_type += 2

    # Fill remaining slots
    while move_type < MOVE_TYPES_PER_FROM:
        ALL_MOVES.append(chess.Move(from_sq, from_sq))
        move_type += 1

# Build lookup tables
MOVE_INDEX_MAP = {mv: idx for idx, mv in enumerate(ALL_MOVES)}
INDEX_MOVE_MAP = {idx: mv for mv, idx in MOVE_INDEX_MAP.items()}


def move_to_index(move: chess.Move) -> int:
    """Return index for given chess.Move, or -1 if not found."""
    return MOVE_INDEX_MAP.get(move, -1)


def index_to_move(index: int) -> chess.Move:
    """Return chess.Move for given index, or None if invalid."""
    return INDEX_MOVE_MAP.get(index, None)


def get_total_move_count() -> int:
    return len(ALL_MOVES)


if __name__ == "__main__":
    print("Total moves:", get_total_move_count())
    test_moves = [
        chess.Move.from_uci("g8f6"),  # knight
        chess.Move.from_uci("b8a6"),  # knight
        chess.Move.from_uci("e7d8q"), # white promotion capture
        chess.Move.from_uci("e7f8r"), # white promotion capture
        chess.Move.from_uci("e1g1"),  # castling
        chess.Move.from_uci("e8c8")   # castling
    ]
    for mv in test_moves:
        print(f"{mv.uci()} → {move_to_index(mv)}")