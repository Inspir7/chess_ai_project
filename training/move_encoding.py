# training/move_encoding.py

import chess

# -------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------

TOTAL_MOVES = 4672   # 64*64 + 4*64  (normal moves + 4 promotion types)

PROMOTION_PIECES = {
    chess.KNIGHT: 0,
    chess.BISHOP: 1,
    chess.ROOK:   2,
    chess.QUEEN:  3,
}

# -------------------------------------------------------
# MOVE → INDEX
# -------------------------------------------------------

def move_to_index(move: chess.Move) -> int:
    """
    Converts a chess.Move into an index (0..4671).
    EXACT SAME FORMULA AS USED IN generate_labeled_data.py.

    Encoding:
        base = 64 * from + to

        If promotion:
            base += 4096 + promo_offset * 64
    """
    from_sq = move.from_square
    to_sq = move.to_square

    index = 64 * from_sq + to_sq  # normal moves

    if move.promotion:
        promo_offset = PROMOTION_PIECES.get(move.promotion, 0)
        index += 4096 + promo_offset * 64

    return index


# -------------------------------------------------------
# INDEX → MOVE  (optional for debugging)
# -------------------------------------------------------

def index_to_move(index: int) -> chess.Move | None:
    """
    Reverse of move_to_index.
    Optional for debugging. Not used by RL/MCTS normally.
    """
    if index < 0 or index >= TOTAL_MOVES:
        return None

    # Promotion moves
    if index >= 4096:
        offset = index - 4096
        promo_type = offset // 64  # 0..3
        to_sq = offset % 64
        from_sq = None  # need to reconstruct

        # reconstruct the correct `from`:
        # we know that base = 64*from + to
        # but here base = to only, so we cannot deduce "from" directly.
        # This function is not required for RL, so we keep minimal logic.

        return chess.Move(None, to_sq, promotion=_promo_from_type(promo_type))

    # Normal move
    from_sq = index // 64
    to_sq = index % 64
    return chess.Move(from_sq, to_sq)


def _promo_from_type(promo_index):
    reverse = {
        0: chess.KNIGHT,
        1: chess.BISHOP,
        2: chess.ROOK,
        3: chess.QUEEN,
    }
    return reverse.get(promo_index, chess.QUEEN)


# -------------------------------------------------------
# PUBLIC API
# -------------------------------------------------------

def get_total_move_count():
    return TOTAL_MOVES


# Quick sanity test
if __name__ == "__main__":
    mv = chess.Move.from_uci("e7e8q")
    idx = move_to_index(mv)
    print("Index:", idx)
    print("Total:", TOTAL_MOVES)