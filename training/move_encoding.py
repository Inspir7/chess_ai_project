# move_encoding.py
import chess

"""
Deterministic fixed-size move encoding with 73 move-types per from-square:
  - index = from_sq * 73 + move_type  (0 <= from_sq < 64, 0 <= move_type < 73)
This yields 64 * 73 = 4672 total distinct entries.

Implementation details:
 - For move_type in [0..72]:
    * to_sq = move_type % 64  (ensures a valid target square 0..63)
    * promotion selection for some move_type values:
        - we reserve a small range of move_types to indicate promotion to
          knight/bishop/rook/queen by adding promotion field.
    * This is deterministic and always produces valid chess.Move objects.
 - Reason: гарантираме фиксиран размер и валидни chess.Move обекти.
"""

# number of move-types per from-square
MOVE_TYPES_PER_FROM = 73
TOTAL_MOVES = 64 * MOVE_TYPES_PER_FROM

# We'll reserve the last 9 move_types for "promotions" variants (arbitrary choice).
# Implementation: for move_type >= 64 use (to_sq, promotion) mapping in cycle of 4 promotions.
PROMO_START = 64  # move_type values >= PROMO_START will include promotions

PROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]  # 4-piece cycle

ALL_MOVES = []

for from_sq in range(64):
    for move_type in range(MOVE_TYPES_PER_FROM):
        # baseline to-square (always valid) — ensures we have unique mapping
        to_sq = move_type % 64

        # promotion decision (deterministic):
        # if move_type >= PROMO_START then assign a promotion piece from cycle
        promotion = None
        if move_type >= PROMO_START:
            promo_idx = (move_type - PROMO_START) % len(PROMO_PIECES)
            promotion = PROMO_PIECES[promo_idx]

        mv = chess.Move(from_sq, to_sq, promotion=promotion)
        ALL_MOVES.append(mv)

# Maps
MOVE_INDEX_MAP = {mv: idx for idx, mv in enumerate(ALL_MOVES)}
INDEX_MOVE_MAP = {idx: mv for mv, idx in MOVE_INDEX_MAP.items()}

def move_to_index(move: chess.Move) -> int:
    """Return index for given chess.Move, or -1 if not found."""
    return MOVE_INDEX_MAP.get(move, -1)

def index_to_move(index: int) -> chess.Move:
    """Return chess.Move for given index, or None if out of range."""
    return INDEX_MOVE_MAP.get(index, None)

def get_total_move_count() -> int:
    """Return total size (should be 4672)."""
    return len(ALL_MOVES)

if __name__ == "__main__":
    print("MOVE_TYPES_PER_FROM =", MOVE_TYPES_PER_FROM)
    print("Total move count:", get_total_move_count())
