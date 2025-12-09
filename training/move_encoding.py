# training/move_encoding.py

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
    chess.ROOK:   2,
    chess.QUEEN:  3,
}

REVERSE_PROMO = {
    0: chess.KNIGHT,
    1: chess.BISHOP,
    2: chess.ROOK,
    3: chess.QUEEN,
}


# ============================================================
# MOVE → INDEX
# (MUST remain identical to generate_labeled_data)
# ============================================================

def move_to_index(move: chess.Move) -> int:
    """
    Convert chess.Move into index 0..4671.
    100% consistent with generate_labeled_data.py
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
# INDEX → MOVE  (Debug use only)
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

        # Cannot reconstruct from-square without board context
        return chess.Move(None, to_sq, promotion=promo)

    # Normal move
    from_sq = index // 64
    to_sq   = index % 64
    return chess.Move(from_sq, to_sq)


# ============================================================
# GEOMETRICAL FLIP FOR DATA AUGMENTATION
# ============================================================

def flip_square(sq: int) -> int:
    """
    Mirror horizontally (a <-> h).
    Works on 0..63.
    """
    rank = sq // 8
    file = sq % 8
    file_flipped = 7 - file
    return rank * 8 + file_flipped


def flip_move_index(idx: int) -> int:
    """
    Mirror a move index over the vertical axis.
    Works for both normal moves and promotions.
    """

    # Promotion moves
    if idx >= 4096:
        offset = idx - 4096
        promo_type = offset // 64  # 0..3
        to_sq = offset % 64

        new_to = flip_square(to_sq)
        return 4096 + promo_type * 64 + new_to

    # Normal move
    from_sq = idx // 64
    to_sq = idx % 64

    new_from = flip_square(from_sq)
    new_to   = flip_square(to_sq)

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