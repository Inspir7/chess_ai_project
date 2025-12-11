import chess
import torch
from models.AlphaZero import AlphaZeroModel
from training.mcts import MCTS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AlphaZeroModel().to(device)
model.load_state_dict(torch.load(
    "/home/presi/projects/chess_ai_project/training/rl/checkpoints/alpha_zero_rl_main.pth",
    map_location=device
))
model.eval()

board = chess.Board()
mcts = MCTS(model, device, simulations=600)

pi = mcts.run(board, move_number=0)

print("Number of moves in pi:", len(pi))
print("Sum of probabilities:", sum(pi.values()))
print("Top moves:")
print(sorted(pi.items(), key=lambda x: -x[1])[:5])
