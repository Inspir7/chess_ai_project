import torch
import chess
from models.policy_head import PolicyHead
from mcts import MCTS

def test_mcts_stability():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyHead().to(device)
    model.eval()  # важно!

    mcts = MCTS(model, device, simulations=50, temperature=1.0)
    board = chess.Board()

    pi = mcts.run(board)

    # Проверки:
    assert isinstance(pi, dict), "MCTS should return a dict"
    assert all(isinstance(k, chess.Move) for k in pi), "Keys should be chess.Move"
    assert all(0 <= v <= 1 for v in pi.values()), "Policy values should be probabilities"
    assert abs(sum(pi.values()) - 1.0) < 1e-4, "Policy probs should sum to 1"

    for move in pi:
        assert move in board.legal_moves, f"Illegal move suggested: {move}"

    print("[TEST] MCTS passed stability test ✅")
