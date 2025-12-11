import os
import sys
import math
import time

import chess
import chess.engine
import torch

# ==============================
# Python path → project root
# ==============================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.AlphaZero import AlphaZeroModel
from utils.mcts_move_selector import mcts_select_move


# ==============================
# Пътища
# ==============================
BASE_DIR = PROJECT_ROOT
TRAINING_DIR = os.path.join(BASE_DIR, "training")

RL_CHECKPOINT = os.path.join(
    TRAINING_DIR, "rl", "checkpoints", "alpha_zero_rl_main.pth"
)

# !! Тук слагаме реалния път до Stockfish (ти вече показа which stockfish → /usr/games/stockfish)
STOCKFISH_PATH = "/usr/games/stockfish"


# ==============================
# Зареждане на модела
# ==============================
def load_model(device: torch.device):
    print(f"[INFO] Loading model from {RL_CHECKPOINT}")
    model = AlphaZeroModel().to(device)
    state_dict = torch.load(RL_CHECKPOINT, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ==============================
# Една партия AlphaZero vs Stockfish
# ==============================
def play_single_game(
    model,
    engine: chess.engine.SimpleEngine,
    device: torch.device,
    stockfish_skill: int = 0,
    simulations: int = 200,
    max_moves: int = 160,
    alphazero_is_white: bool = True,
):
    """
    Връща:
      result_str: "1-0" / "0-1" / "1/2-1/2"
      score: резултат от гледна точка на AlphaZero (1, 0.5, 0)
    """
    board = chess.Board()

    # Настройка на Stockfish
    engine.configure({"Skill Level": stockfish_skill})

    ply = 0

    while not board.is_game_over() and ply < max_moves:
        to_move_is_white = board.turn == chess.WHITE
        alphazero_to_move = (to_move_is_white and alphazero_is_white) or (
            (not to_move_is_white) and (not alphazero_is_white)
        )

        if alphazero_to_move:
            # Ход на AlphaZero чрез MCTS
            move = mcts_select_move(
                model=model,
                board=board,
                device=device,
                simulations=simulations,
                ply=ply,
            )

            if move is None:
                # Няма ход → губим
                break

            board.push(move)
        else:
            # Ход на Stockfish
            try:
                # Леко ограничение – да не мисли твърде дълго
                # depth=8 е разумен компромис за skill 0–1
                result = engine.play(board, chess.engine.Limit(depth=8))
                if result.move is None:
                    break
                board.push(result.move)
            except chess.engine.EngineTerminatedError:
                print("[ERROR] Stockfish engine terminated unexpectedly.")
                break
            except chess.engine.EngineError as e:
                print(f"[ERROR] Stockfish error: {e}")
                break

        ply += 1

    # Финален резултат
    if board.is_game_over():
        res = board.result()
    else:
        # Ако стигнем max_moves без резултат → реми
        res = "1/2-1/2"

    # Пресмятаме резултат от гледна точка на AlphaZero
    if res == "1-0":
        score_white = 1.0
    elif res == "0-1":
        score_white = 0.0
    else:
        score_white = 0.5

    if alphazero_is_white:
        score_az = score_white
    else:
        score_az = 1.0 - score_white if res in ["1-0", "0-1"] else 0.5

    return res, score_az


# ==============================
# Серия партии на дадено ниво
# ==============================
def evaluate_vs_stockfish_level(
    model,
    device: torch.device,
    stockfish_skill: int,
    num_games: int = 10,
    simulations: int = 200,
    max_moves: int = 160,
):
    """
    Играем num_games партии срещу Stockfish на даден skill.
    Редуваме цветове (бяла/черна) за AlphaZero.
    Връща:
      avg_score: среден резултат от гледна точка на AlphaZero
      elo_diff: приблизителна Elo разлика (AlphaZero – Stockfish)
    """
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Stockfish binary not found at: {STOCKFISH_PATH}")
        sys.exit(1)

    scores = []

    try:
        for game_idx in range(1, num_games + 1):
            alphazero_is_white = (game_idx % 2 == 1)

            print(
                f"Game {game_idx}/{num_games} "
                f"(AZ as {'White' if alphazero_is_white else 'Black'}): ",
                end="",
                flush=True,
            )

            res, score = play_single_game(
                model=model,
                engine=engine,
                device=device,
                stockfish_skill=stockfish_skill,
                simulations=simulations,
                max_moves=max_moves,
                alphazero_is_white=alphazero_is_white,
            )

            scores.append(score)
            outcome_char = "W" if score == 1.0 else ("D" if score == 0.5 else "L")
            print(f"{outcome_char} ({res})")

    finally:
        engine.quit()

    if not scores:
        return 0.0, None

    avg_score = sum(scores) / len(scores)

    # Приблизителна Elo разлика:
    # p = score за AlphaZero; elo_diff = -400 * log10(1/p - 1)
    # (ако p=0 или 1 → clamp-ваме)
    p = max(0.01, min(0.99, avg_score))
    elo_diff = -400.0 * math.log10(1.0 / p - 1.0)

    return avg_score, elo_diff


# ==============================
# MAIN
# ==============================
def main():
    print("\n==============================")
    print(" AlphaZero vs Stockfish Evaluation")
    print("==============================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Зареждаме RL модела
    model = load_model(device)

    # Конфигурация
    simulations = 200     # MCTS simulations per move
    max_moves = 160       # лимит на полуходове
    num_games_per_level = 10

    # Нива на Stockfish за тест – можеш да добавиш [2,3,...]
    levels = [0, 1]

    for lvl in levels:
        print(f"\n=== Stockfish Skill {lvl} ===")
        avg_score, elo_diff = evaluate_vs_stockfish_level(
            model=model,
            device=device,
            stockfish_skill=lvl,
            num_games=num_games_per_level,
            simulations=simulations,
            max_moves=max_moves,
        )

        print(f"→ Avg score vs skill {lvl}: {avg_score:.3f}")
        if elo_diff is None:
            print("→ Elo difference: N/A (no games)")
        else:
            print(f"→ Approx ELO difference (AZ - SF): {elo_diff:.1f}")


if __name__ == "__main__":
    main()