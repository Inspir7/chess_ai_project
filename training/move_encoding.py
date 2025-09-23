import chess

def generate_all_possible_uci_moves():
    squares = [chess.square_name(i) for i in range(64)]
    promotions = ['q', 'r', 'b', 'n']
    moves = []

    for from_sq in squares:
        for to_sq in squares:
            if from_sq == to_sq:
                continue  # пропусни ходове като a1a1

            move = from_sq + to_sq
            moves.append(move)

            # промоции от черните
            if from_sq[1] == '7' and to_sq[1] == '8':
                for promo in promotions:
                    moves.append(move + promo)

            # промоции от белите
            if from_sq[1] == '2' and to_sq[1] == '1':
                for promo in promotions:
                    moves.append(move + promo)

    return moves


# Списък с всички възможни ходове като обекти от тип `chess.Move`
ALL_UCI_MOVES = generate_all_possible_uci_moves()
ALL_CHESS_MOVES = [chess.Move.from_uci(mv) for mv in ALL_UCI_MOVES]

# Речници за кодиране и декодиране
MOVE_INDEX_MAP = {move: idx for idx, move in enumerate(ALL_CHESS_MOVES)}
INDEX_MOVE_MAP = {idx: move for move, idx in MOVE_INDEX_MAP.items()}


def move_to_index(move: chess.Move) -> int:
    """Връща индекса на даден ход"""
    return MOVE_INDEX_MAP.get(move, -1)


def index_to_move(index: int) -> chess.Move:
    """Връща chess.Move обект по индекс"""
    return INDEX_MOVE_MAP.get(index, None)


def get_total_move_count() -> int:
    return len(MOVE_INDEX_MAP)
