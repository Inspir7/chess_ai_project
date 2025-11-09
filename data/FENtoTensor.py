# ne e nujno
import chess
import numpy as np

piece_to_index = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}

def fen_to_tensor(fen: str) -> np.ndarray:
    """
    Конвертира FEN низ в 8x8x15 тензор за AlphaZero модел.
    Канали:
    0-5: бели фигури P,N,B,R,Q,K
    6-11: черни фигури p,n,b,r,q,k
    12: текущ играч (1=бял, 0=черен)
    13: нормализиран номер на хода (0-1)
    14: фаза на играта (0=дебют, 1=миттелшпил, 2=ендшпил)
    """
    board = chess.Board(fen)
    tensor = np.zeros((8, 8, 15), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - chess.square_rank(square)  # 0-index top=rank7
            col = chess.square_file(square)
            idx = piece_to_index[piece.symbol()]
            tensor[row, col, idx] = 1.0

    # Текущ играч
    tensor[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0

    # Номер на хода нормализиран
    tensor[:, :, 13] = board.fullmove_number / 100.0

    # Фаза на играта (примерна груба оценка)
    total_material = sum([piece.piece_type for piece in board.piece_map().values()])
    if total_material > 40:
        phase = 0.0  # дебют
    elif total_material > 20:
        phase = 1.0  # миттелшпил
    else:
        phase = 2.0  # ендшпил
    tensor[:, :, 14] = phase

    return tensor
