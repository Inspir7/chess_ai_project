import chess
import torch

from training.mcts import MCTS
from data.generate_labeled_data import fen_to_tensor


def _encode_board(board: chess.Board, device: torch.device) -> torch.Tensor:
    """
    Превръща FEN-а на дъската в вход за модела: shape (1, C, 8, 8).
    Използва същия encode, както при training.
    """
    tens = fen_to_tensor(board.fen())
    x = torch.tensor(tens, dtype=torch.float32)

    # Ако е (8, 8, C) → правим го (C, 8, 8)
    if x.ndim == 3 and x.shape[0] == 8 and x.shape[1] == 8:
        # HWC → CHW
        x = x.permute(2, 0, 1)
    else:
        x = x.view(15, 8, 8)

    x = x.unsqueeze(0).to(device)
    return x


def mcts_select_move(
    model,
    board: chess.Board,
    device: torch.device,
    simulations: int = 200,
    ply: int = 0,
):
    """
    Връща ход според MCTS, максимизиращ вероятността (без шум, без температура).
    Това е отделено от self_play, за да не пипаме training логиката.

    Използване:
        move = mcts_select_move(model, board, device, simulations=200, ply=current_ply)
    """
    model.eval()

    # MCTS както в self_play, само че с подаден ply (move_number)
    mcts = MCTS(model, device, simulations=simulations)

    # Изпълняваме търсене
    pi_dict = mcts.run(board, move_number=ply)

    if not pi_dict:
        # Ако нещо се обърка, връщаме първия легален
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        return legal_moves[0]

    # Вземаме хода с максимална вероятност (argmax по π)
    best_move = max(pi_dict.items(), key=lambda kv: kv[1])[0]
    return best_move


def evaluate_position_value(model, board: chess.Board, device: torch.device) -> float:
    """
    Дава оценката на value главата на модела за дадена позиция.
    Полезно за диагностика и дебъг, не се ползва в самия evaluator.
    """
    model.eval()

    x = _encode_board(board, device)

    with torch.no_grad():
        _, value = model(x)

    return float(value.item())