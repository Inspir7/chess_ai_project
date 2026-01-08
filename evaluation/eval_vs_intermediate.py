import sys
import os
import random
import chess
import torch
import numpy as np

# ==========================
# 1. НАСТРОЙКА НА ПЪТИЩАТА
# ==========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.AlphaZero import AlphaZeroModel
from training.mcts import MCTS

# ==========================
# 2. КОНФИГУРАЦИЯ
# ==========================
MODEL_PATH = os.path.join(PROJECT_ROOT, "training/rl/checkpoints/alpha_zero_rl_checkpoint_final.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Настройки за теста
AI_SIMULATIONS = 2000
ADJUDICATION_LIMIT = 9  # Ако водиш с 9 точки (Дама), печелиш веднага
MAX_MOVES = 150  # Максимална дължина на играта

# Стойности на фигурите за Material Bot и оценката
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 1000
}


# ==========================
# 3. ПОМОЩНИ ФУНКЦИИ
# ==========================

def get_material_score(board, color):
    """
    Изчислява материалния баланс от гледна точка на даден цвят.
    Връща положително число, ако 'color' води.
    """
    score = 0
    for pt, val in PIECE_VALUES.items():
        score += len(board.pieces(pt, color)) * val
        score -= len(board.pieces(pt, not color)) * val
    return score


def get_weak_bot_move(board):
    """
    Material Bot (Greedy):
    - Гледа 1 ход напред.
    - Взима фигура, ако може.
    - Ако няма взимане, играе случайно.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    best_moves = []
    best_score = -float('inf')
    turn = board.turn

    for move in legal_moves:
        board.push(move)

        # Ако е мат - това е най-доброто!
        if board.is_checkmate():
            score = 99999
        else:
            score = get_material_score(board, turn)

        board.pop()

        if score > best_score:
            best_score = score
            best_moves = [move]
        elif score == best_score:
            best_moves.append(move)

    return random.choice(best_moves)


def get_ai_move(model, board, simulations=2000):
    """
    AlphaZero AI move selector
    """
    # Създаваме MCTS с 0 шум (детерминистичен за тест)
    mcts = MCTS(
        model,
        DEVICE,
        simulations=simulations,
        dirichlet_epsilon=0.0,
        c_puct=1.5
    )

    try:
        # Лъжем го, че е ход 100, за да е ниска температурата вътрешно
        pi = mcts.run(board, move_number=100)

        moves = list(pi.keys())
        probs = list(pi.values())

        if not moves: return None

        # Винаги избираме най-вероятния ход (Argmax)
        return moves[np.argmax(probs)]
    except Exception as e:
        print(f"Error in MCTS: {e}")
        return None


# ==========================
# 4. ИГРАЛЕН ЦИКЪЛ
# ==========================

def play_game(model, ai_color):
    board = chess.Board()
    ai_color_str = "White" if ai_color == chess.WHITE else "Black"
    opponent_str = "Material Bot (Elo ~350)"

    print(f"\n⚔️  AI ({ai_color_str}) vs {opponent_str}")
    print(f"   Settings: Sims={AI_SIMULATIONS}, Adjudication=+{ADJUDICATION_LIMIT}")

    moves = 0
    while not board.is_game_over() and moves < MAX_MOVES:

        # --- ADJUDICATION (Служебна победа) ---
        # Ако сме минали дебюта (30 хода) и някой води с много материал
        if moves > 30:
            diff = get_material_score(board, ai_color)

            # Ако AI води с повече от 9 точки (Дама)
            if diff >= ADJUDICATION_LIMIT:
                print(f"      🏆 ADJUDICATION: AI wins by huge material advantage (+{diff})!")
                return 1.0

            # Ако AI губи с повече от 9 точки
            elif diff <= -ADJUDICATION_LIMIT:
                print(f"      💀 ADJUDICATION: AI loses by huge material deficit ({diff}).")
                return 0.0

        # --- ИЗБОР НА ХОД ---
        if board.turn == ai_color:
            # AI играе
            move = get_ai_move(model, board, simulations=AI_SIMULATIONS)
        else:
            # Ботът играе
            move = get_weak_bot_move(board)

        # Проверка за валидност
        if move is None or move not in board.legal_moves:
            print("❌ Game Over: No legal move found or crash.")
            break

        board.push(move)
        moves += 1

        # Лог на всеки 10 хода
        if moves % 10 == 0:
            diff = get_material_score(board, ai_color)
            print(f"   Move {moves} | Material Diff: {diff}")

    # --- КРАЙ НА ИГРАТА ---
    res = board.result()
    print(f"🏁 Final Result: {res} (Moves: {moves})")

    if res == "1-0":
        return 1.0 if ai_color == chess.WHITE else 0.0
    elif res == "0-1":
        return 1.0 if ai_color == chess.BLACK else 0.0
    else:
        # При реми:
        # Ако AI има предимство, пак го броим за частичен успех (0.5)
        # В дипломната можеш да го наречеш "Draw"
        return 0.5


# ==========================
# 5. MAIN (Скрипт за стартиране)
# ==========================
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        exit()

    print(f"🤖 Loading AlphaZero Model...")
    model = AlphaZeroModel().to(DEVICE)

    try:
        # Опит за зареждане на чист state_dict
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except:
        # Fallback: ако е пълен checkpoint (с optimizer и т.н.)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)  # Опит директно

    model.eval()
    print("✅ Model loaded successfully.")

    # Игра 1: AI с Белите
    print("\n--- GAME 1: AI is WHITE ---")
    score_w = play_game(model, chess.WHITE)

    # Игра 2: AI с Черните
    print("\n--- GAME 2: AI is BLACK ---")
    score_b = play_game(model, chess.BLACK)

    total_score = score_w + score_b

    print("\n==================================")
    print(f"📊 FINAL SCORE vs Material Bot: {total_score}/2.0")
    print("==================================")

    if total_score >= 1.5:
        print("🚀 SUCCESS! The model is clearly stronger than simple material play.")
        print("Estimated Elo: > 400")
    elif total_score >= 1.0:
        print("⚠️ DECENT. Matches material play.")
        print("Estimated Elo: ~300-350")
    else:
        print("❌ NEEDS IMPROVEMENT. Losing tactically.")
        print("Estimated Elo: < 300")