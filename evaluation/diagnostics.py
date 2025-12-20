import torch
import chess
import sys
import os
import numpy as np

# Добавяме пътя, за да намерим training.mcts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.AlphaZero import AlphaZeroModel
from training.mcts import MCTS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/presi/projects/chess_ai_project/training/alpha_zero_supervised_final.pth"


def debug_position(model, fen, description):
    print(f"\n===================================================")
    print(f"🔍 ТЕСТ: {description}")
    print(f"FEN: {fen}")
    board = chess.Board(fen)

    # Визуализация (за да видим кой е на ход)
    print(f"На ход: {'WHITE' if board.turn == chess.WHITE else 'BLACK'}")

    # 1. Пускаме MCTS (точно както в self_play)
    mcts = MCTS(model, DEVICE, simulations=100)  # 100 е достатъчно за мат в 1
    pi_dict = mcts.run(board, move_number=0)

    if not pi_dict:
        print("❌ ГРЕШКА: MCTS не върна никакви ходове!")
        return

    # 2. Виждаме какво мисли моделът
    # Сортираме ходовете по посещения (най-добрите са първи)
    sorted_moves = sorted(pi_dict.items(), key=lambda x: x[1], reverse=True)

    print(f"--- 📊 Какво иска да играе AI? ---")
    best_move = sorted_moves[0][0]

    for i, (move, prob) in enumerate(sorted_moves[:3]):
        print(f"{i + 1}. {move} -> Увереност: {prob:.4f}")

    # 3. Проверка дали е намерил мата
    board.push(best_move)
    if board.is_checkmate():
        print(f"\n✅ УСПЕХ: Намери мат! ({best_move})")
    else:
        print(f"\n⚠️ ПРОВАЛ: Пропусна мат в 1 ход. Изигра {best_move}")


def main():
    if not os.path.exists(MODEL_PATH):
        print("❌ Не намирам модела!")
        return

    print("Loading model...")
    model = AlphaZeroModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # ТЕСТ 1: Класически мат с Топ (Бели на ход)
    # Цар на e1, Топ на a1. Черен цар на a8. Ra1-a8 е мат.
    fen_white_mate = "k7/8/8/8/8/8/8/R3K3 w - - 0 1"
    debug_position(model, fen_white_mate, "Мат в 1 (БЕЛИ)")

    # ТЕСТ 2: Класически мат с Топ (ЧЕРНИ на ход) <--- ТУК ЩЕ ЛЪСНЕ ПРОБЛЕМА С ОГЛЕДАЛОТО
    # Бял цар на h1. Черен цар на e8, Топ на h8. ...Rh8-h1 е мат.
    fen_black_mate = "4k3/8/8/8/8/8/8/7K b - - 0 1"
    debug_position(model, fen_black_mate, "Мат в 1 (ЧЕРНИ)")


if __name__ == "__main__":
    main()