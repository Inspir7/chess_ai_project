import pygame
import chess
import torch
import os
import matplotlib.pyplot as plt
from torch import optim

from training.mcts import MCTS
from models.AlphaZero import AlphaZeroModel
from utils.replay_buffer import save_replay_buffer, load_replay_buffer
from utils.promotion import show_promotion_menu
from training.train_on_batch import train_on_batch
from training.move_encoding import move_to_index, get_total_move_count
from training.generate_labeled_data import fen_to_tensor

pygame.init()

WIDTH, HEIGHT = 640, 640
SQUARE_SIZE = WIDTH // 8
INFO_PANEL_WIDTH = 180
WHITE = (255, 255, 255)
DARK_PINK = (199, 21, 133)

screen = pygame.display.set_mode((WIDTH + INFO_PANEL_WIDTH, HEIGHT))
pygame.display.set_caption("Barbie Chess")
font = pygame.font.SysFont("Arial", 20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlphaZeroModel().to(device)
model_path = "/home/presi/projects/chess_ai_project/training/alpha_zero_supervised.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
optimizer = optim.Adam(model.parameters(), lr=0.001)

ai_agent = MCTS(model=model, device=device, simulations=100)
board = chess.Board()
mode = "human_vs_ai"
replay_buffer = load_replay_buffer("/home/presi/projects/chess_ai_project/utils/replays/replay_buffer.pkl")

piece_images = {symbol: pygame.transform.scale(pygame.image.load(f"assets/{color}_{symbol.upper()}.png"), (SQUARE_SIZE, SQUARE_SIZE))
                for symbol, color in zip("prnbqkPRNBQK", ["b"] * 6 + ["w"] * 6)}

USE_BOOTSTRAPPED_REWARDS = False
USE_MATERIAL_REWARD = False
loss_history = []
game_counter = 0

def draw_board():
    colors = [pygame.Color("white"), pygame.Color("lightpink")]
    for rank in range(8):
        for file in range(8):
            color = colors[(rank + file) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(file*SQUARE_SIZE, rank*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_pieces(dragging_piece=None, dragging_pos=None):
    for square in chess.SQUARES:
        if dragging_piece and square == dragging_piece[0]:
            continue
        piece = board.piece_at(square)
        if piece:
            file = chess.square_file(square)
            rank = 7 - chess.square_rank(square)
            screen.blit(piece_images[piece.symbol()],
                        (file * SQUARE_SIZE, rank * SQUARE_SIZE))
    if dragging_piece and dragging_pos:
        screen.blit(piece_images[dragging_piece[1].symbol()],
                    (dragging_pos[0] - SQUARE_SIZE // 2, dragging_pos[1] - SQUARE_SIZE // 2))

def draw_info_panel():
    panel_x = WIDTH
    pygame.draw.rect(screen, DARK_PINK, pygame.Rect(panel_x, 0, INFO_PANEL_WIDTH, HEIGHT))
    btn = pygame.Rect(panel_x + 10, 200, 160, 40)
    toggle_btn = pygame.Rect(panel_x + 10, 300, 160, 40)
    pygame.draw.rect(screen, (255, 255, 255), btn)
    pygame.draw.rect(screen, DARK_PINK, toggle_btn)
    screen.blit(font.render("Нова игра", True, (0, 0, 0)), (btn.x + 20, btn.y + 10))
    screen.blit(font.render("Смени режим", True, WHITE), (toggle_btn.x + 10, toggle_btn.y + 10))
    return btn, toggle_btn

def train_from_replay_buffer():
    if len(replay_buffer) >= 20:
        print("[Обучение] Стартиране от buffer с дължина:", len(replay_buffer))
        final_examples = []

        for fen, game_result in replay_buffer[-100:]:
            if game_result is None:
                continue

            board_copy = chess.Board(fen)

            if board_copy.is_game_over():
                continue  # skip non-legal examples

            tensor = fen_to_tensor(board_copy.fen())
            if tensor.shape != (8, 8, 15):
                tensor = tensor.transpose(2, 0, 1)
            state_tensor = torch.tensor(tensor, dtype=torch.float32).permute(2, 0, 1)

            pi_dict, _ = ai_agent._model_eval(board_copy)
            pi = torch.zeros(get_total_move_count())
            for mv, prob in pi_dict.items():
                idx = move_to_index(mv)
                if idx >= 0:
                    pi[idx] = prob

            v = game_result if board_copy.turn == chess.WHITE else -game_result
            final_examples.append((state_tensor, pi, v))

        if not final_examples:
            print("[Обучение] Пропуснато - няма валидни примери.")
            return

        loss_policy, loss_value = train_on_batch(model, final_examples, device)
        loss_history.append((loss_policy + loss_value))
        print(f"[Обучение] Загуба: policy={loss_policy:.4f}, value={loss_value:.4f}")

def run_multiple_self_play_games(n=10):
    global board, replay_buffer
    for i in range(n):
        print(f"[Self-Play] Стартиране на игра {i+1}/{n}")
        board.reset()
        replay_buffer = []
        while not board.is_game_over():
            move = ai_agent.select_move(board)
            board.push(move)
            replay_buffer.append((board.fen(), None))

        result_str = board.result()
        if result_str == '1-0':
            final_result = 1.0
        elif result_str == '0-1':
            final_result = -1.0
        else:
            final_result = 0.0

        replay_buffer = [(fen, final_result) for fen, _ in replay_buffer]
        train_from_replay_buffer()
    plot_training_progress()
    print("[Self-Play] Всички игри приключиха.")

def plot_training_progress():
    if loss_history:
        plt.figure(figsize=(6, 3))
        plt.plot(loss_history, label="Total Loss", color="deeppink")
        plt.xlabel("Игри")
        plt.ylabel("Загуба")
        plt.title("Прогрес на обучението")
        plt.legend()
        plt.tight_layout()
        plt.savefig("training_progress.png")
        plt.close()

def main():
    global mode, board, replay_buffer
    running = True
    dragging = False
    drag_start_square = None
    dragging_piece = None
    training_done = False  # flag

    while running:
        draw_board()
        draw_pieces((drag_start_square, dragging_piece) if dragging_piece else None, pygame.mouse.get_pos() if dragging else None)
        btn_rect, toggle_btn_rect = draw_info_panel()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_replay_buffer(replay_buffer)
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    run_multiple_self_play_games(10)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if btn_rect.collidepoint(x, y):
                    board.reset()
                    training_done = False  # glag reset
                    train_from_replay_buffer()
                    replay_buffer = []
                    continue
                elif toggle_btn_rect.collidepoint(x, y):
                    mode = "self_play" if mode == "human_vs_ai" else "human_vs_ai"
                    print("Режим сменен на:", mode)
                    continue

                if x < WIDTH and y < HEIGHT:
                    file = x // SQUARE_SIZE
                    rank = 7 - (y // SQUARE_SIZE)
                    square = chess.square(file, rank)
                    piece = board.piece_at(square)
                    if piece and piece.color == board.turn:
                        dragging = True
                        drag_start_square = square
                        dragging_piece = piece

            elif event.type == pygame.MOUSEBUTTONUP and dragging:
                x, y = event.pos
                file = x // SQUARE_SIZE
                rank = 7 - (y // SQUARE_SIZE)
                drop_square = chess.square(file, rank)
                move = chess.Move(drag_start_square, drop_square)
                promotion_move = chess.Move(drag_start_square, drop_square, promotion=chess.QUEEN)

                if move in board.legal_moves:
                    board.push(move)
                    replay_buffer.append((board.fen(), None))
                    print(f"Ход: {move.uci()}")

                elif promotion_move in board.legal_moves:
                    is_white = board.turn
                    promotion_piece = show_promotion_menu(screen, board, drop_square, is_white)
                    move = chess.Move(drag_start_square, drop_square, promotion=promotion_piece)

                    if move in board.legal_moves:
                        board.push(move)
                        replay_buffer.append((board.fen(), None))
                        print(f"Ход (промоция): {move.uci()}")

                else:
                    print(f"Невалиден ход: {move.uci()}")

                if move in board.move_stack and mode == "human_vs_ai" and not board.is_game_over():
                    ai_move = ai_agent.select_move(board)
                    board.push(ai_move)
                    replay_buffer.append((board.fen(), None))
                    print(f"AI ход: {ai_move.uci()}")

                dragging = False
                dragging_piece = None
                drag_start_square = None

                if board.is_game_over() and not training_done:
                    if replay_buffer and replay_buffer[0][1] is None:
                        result_str = board.result()
                        if result_str == '1-0':
                            final_result = 1.0
                        elif result_str == '0-1':
                            final_result = -1.0
                        else:
                            final_result = 0.0
                        replay_buffer = [(fen, final_result) for fen, _ in replay_buffer]
                    train_from_replay_buffer()
                    training_done = True

        if mode == "self_play" and not board.is_game_over():
            pygame.time.wait(500)
            move = ai_agent.select_move(board)
            board.push(move)
            replay_buffer.append((board.fen(), None))
            print(f"Self-play ход: {move.uci()}")

        if board.is_game_over() and not training_done:
            if replay_buffer and replay_buffer[0][1] is None:
                result_str = board.result()
                if result_str == '1-0':
                    final_result = 1.0
                elif result_str == '0-1':
                    final_result = -1.0
                else:
                    final_result = 0.0
                replay_buffer = [(fen, final_result) for fen, _ in replay_buffer]
            train_from_replay_buffer()
            training_done = True

    pygame.quit()

if __name__ == "__main__":
    main()
