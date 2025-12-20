import torch
import os
from models.AlphaZero import AlphaZeroModel
from training.self_play import play_episode

# === НАСТРОЙКИ ===
MODEL_1_PATH = "/home/presi/projects/chess_ai_project/training/rl/checkpoints/alpha_zero_rl_checkpoint_final.pth"  # Новият шампион
MODEL_2_PATH = "/home/presi/projects/chess_ai_project/training/alpha_zero_supervised_STAGE3.pth"  # Хищникът (от Етап 2/3)
# Ако нямаш STAGE3, ползвай STAGE4 или просто alpha_zero_supervised_ORIGINAL.pth

GAMES = 10  # Брой игри
SIMS = 400  # Симулации (сила на мисълта)
TEMP = 0.1  # Почти нулева температура (играят най-силно)


def load_model(path, device):
    model = AlphaZeroModel().to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"[INFO] Loaded: {path}")
    except:
        print(f"[ERROR] Could not load {path}")
    model.eval()
    return model


def run_match():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== ⚔️ STARTING ARENA MATCH ⚔️ ===")
    print(f"Gladiator 1 (NEW): {MODEL_1_PATH}")
    print(f"Gladiator 2 (OLD): {MODEL_2_PATH}")
    print(f"Simulations: {SIMS} | Device: {device}")

    p1 = load_model(MODEL_1_PATH, device)
    p2 = load_model(MODEL_2_PATH, device)

    score_p1 = 0
    score_p2 = 0
    draws = 0

    for i in range(1, GAMES + 1):
        # Редуваме цветовете
        # Ако i е нечетно: P1 е Бял, P2 е Черен
        # Ако i е четно: P2 е Бял, P1 е Черен
        if i % 2 != 0:
            white_model, black_model = p1, p2
            p1_color = "White"
        else:
            white_model, black_model = p2, p1
            p1_color = "Black"

        print(f"\nGame {i}/{GAMES} (New Model is {p1_color})...")

        # Тук използваме play_episode, но трябва леко да се адаптира логиката,
        # защото play_episode по подразбиране ползва 'frozen_model' само за evaluation.
        # За по-лесно, ще ползваме текущата логика, където model играе срещу frozen.

        # ВАЖНО: В твоя self_play.py 'model' е винаги този, който се учи (Player),
        # а 'frozen' е опонента.

        examples, result, length = play_episode(
            model=white_model,
            frozen_model=black_model,
            device=device,
            simulations=SIMS,
            base_temperature=TEMP,
            verbose=False
        )

        # Result е от гледна точка на White (1-0, 0-1, 1/2-1/2)
        print(f"  -> Result: {result} ({length} moves)")

        if result.startswith("1-0"):
            if p1_color == "White":
                score_p1 += 1
            else:
                score_p2 += 1
        elif result.startswith("0-1"):
            if p1_color == "Black":
                score_p1 += 1
            else:
                score_p2 += 1
        else:
            draws += 1

        print(f"  STATUS: New: {score_p1} | Old: {score_p2} | Draws: {draws}")

    print("\n=== 🏁 FINAL SCORE 🏁 ===")
    print(f"NEW MODEL: {score_p1}")
    print(f"OLD MODEL: {score_p2}")
    print(f"DRAWS    : {draws}")


if __name__ == "__main__":
    run_match()