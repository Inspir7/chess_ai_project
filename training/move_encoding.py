import chess

# ============================================================
# CONSTANTS
# AlphaZero-style flat policy: 4672 entries
#   0..4095    = 64*64 normal moves
#   4096..4159 = promotions to Knight (64)
#   4160..4223 = promotions to Bishop (64)
#   4224..4287 = promotions to Rook   (64)
#   4288..4351 = promotions to Queen  (64)
# TOTAL = 4096 + 4*64 = 4672
# ============================================================

TOTAL_MOVES = 4672

PROMOTION_MAP = {
    chess.KNIGHT: 0,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 3,
}

REVERSE_PROMO = {
    0: chess.KNIGHT,
    1: chess.BISHOP,
    2: chess.ROOK,
    3: chess.QUEEN,
}


# ============================================================
# MOVE -> INDEX
# ============================================================

def move_to_index(move: chess.Move) -> int:
    """
    Convert chess.Move into index 0..4671.
    """
    from_sq = move.from_square
    to_sq = move.to_square

    index = 64 * from_sq + to_sq  # normal move

    # Promotion
    if move.promotion:
        promo_type = PROMOTION_MAP.get(move.promotion, 3)
        index = 4096 + promo_type * 64 + to_sq

    return index


# ============================================================
# INDEX -> MOVE  (Debug use only)
# ============================================================

def index_to_move(index: int) -> chess.Move | None:
    if index < 0 or index >= TOTAL_MOVES:
        return None

    # Promotion zone
    if index >= 4096:
        offset = index - 4096
        promo_type = offset // 64
        to_sq = offset % 64
        promo = REVERSE_PROMO.get(promo_type, chess.QUEEN)
        # Cannot reconstruct from-square exactly without board context in this scheme
        # but for flipping logic we don't strictly need it if we calculate mathematically
        return None

        # Normal move
    from_sq = index // 64
    to_sq = index % 64
    return chess.Move(from_sq, to_sq)


# ============================================================
# 180-DEGREE FLIP (CANONICAL)
# ============================================================

def flip_move_index(idx: int) -> int:
    """
    Mirror a move index over the CENTER (180 degrees rotation).
    This matches the logic: board.transform(chess.flip_vertical).transform(chess.flip_horizontal)

    Logic:
    Square 0 (a1) <-> Square 63 (h8)
    Formula: new_sq = 63 - old_sq
    """

    # ---------------------------
    # A) PROMOTION MOVES
    # ---------------------------
    if idx >= 4096:
        offset = idx - 4096
        promo_type = offset // 64  # 0..3 (Knight..Queen)
        to_sq = offset % 64

        # 180 rotation for destination square
        new_to = 63 - to_sq

        # Reconstruct index
        return 4096 + promo_type * 64 + new_to

    # ---------------------------
    # B) NORMAL MOVES
    # ---------------------------
    from_sq = idx // 64
    to_sq = idx % 64

    # 180 rotation
    new_from = 63 - from_sq
    new_to = 63 - to_sq

    return new_from * 64 + new_to


# ============================================================
# PUBLIC API
# ============================================================

def get_total_move_count() -> int:
    return TOTAL_MOVES

# Quick test
if __name__ == "__main__":
    print("TOTAL =", TOTAL_MOVES)
    mv = chess.Move.from_uci("e7e8q")
    print("e7e8q index:", move_to_index(mv))