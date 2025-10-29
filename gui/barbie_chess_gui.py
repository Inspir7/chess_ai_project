import pygame
import sys
import chess
import torch
import numpy as np
import threading
import time

from utils.chess_utils import (
    initial_board,
    draw_board,
    get_square_from_mouse,
    make_move,
    choose_promotion,
    SQUARE_SIZE,
)
from training.mcts import MCTS
from models.AlphaZero import AlphaZeroModel
from data.FENtoTensor import fen_to_tensor
from training.buffer import ReplayBuffer

# --- GUI лог ---
log_messages = []

def log_to_gui(message):
    log_messages.append(message)
    if len(log_messages) > 15:
        log_messages.pop(0)
    print(message)

# --- Pygame setup ---
pygame.init()
screen = pygame.display.set_mode((SQUARE_SIZE*8+250, SQUARE_SIZE*8+120))
pygame.display.set_caption("Barbie Chess GUI")
font = pygame.font.SysFont("dejavusans", 28)
clock = pygame.time.Clock()

# --- Игрови състояния ---
board = initial_board()
selected_square = None
last_move = None
move_number = 1
mode = "Human vs Human"

# --- Бутони ---
BUTTON_COLOR = (255,105,180)
BUTTON_TEXT_COLOR = (255,255,255)
button_font = pygame.font.SysFont("dejavusans", 28)
buttons = {
    "New Game": pygame.Rect(20, SQUARE_SIZE*8+50, 180,50),
    "Toggle Mode": pygame.Rect(220, SQUARE_SIZE*8+50, 220,50)
}

# --- AI модел ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlphaZeroModel().to(device)
model.load_state_dict(torch.load("/home/presi/projects/chess_ai_project/training/alpha_zero_supervised.pth", map_location=device))
model.eval()

buffer = ReplayBuffer(max_size=5000)
buffer_lock = threading.Lock()   # <-- lock за thread-safety
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
mcts_ai = MCTS(model=model, device=device, simulations=50, c_puct=1.0, temperature=1.0)

# --- RL live-log променливи ---
rl_pi = None
rl_value = 0.0
rl_loss = 0.0
rl_buffer_size = 0

# --- Helper constants ---
TOTAL_MOVES = 64*64

def move_to_index(move):
    return move.from_square * 64 + move.to_square

# --- Draw functions ---
def draw_status_bar():
    pygame.draw.rect(screen, (245,225,240), (0, SQUARE_SIZE*8, SQUARE_SIZE*8, 40))
    status_text = f"{mode} | {'Game Over!' if board.is_game_over() else 'Your turn'}"
    text_surface = font.render(status_text, True, (80,0,80))
    screen.blit(text_surface, (20, SQUARE_SIZE*8+10))

def draw_buttons():
    for label, rect in buttons.items():
        pygame.draw.rect(screen, BUTTON_COLOR, rect, border_radius=12)
        txt = button_font.render(label, True, BUTTON_TEXT_COLOR)
        txt_rect = txt.get_rect(center=rect.center)
        screen.blit(txt, txt_rect)

def draw_live_stats(screen, pi=None, value=0.0, move_number=1, loss=0.0):
    pygame.draw.rect(screen, (245,245,245), (SQUARE_SIZE*8,0,250,SQUARE_SIZE*8))
    font_stats = pygame.font.SysFont("dejavusans", 20)
    y = 20
    screen.blit(font_stats.render(f"Move #{move_number}", True, (0,0,0)), (SQUARE_SIZE*8+10,y))
    y+=30
    screen.blit(font_stats.render(f"Value: {rl_value:.2f}", True, (0,0,0)), (SQUARE_SIZE*8+10,y))
    y+=30
    screen.blit(font_stats.render(f"Loss: {rl_loss:.4f}", True, (0,0,0)), (SQUARE_SIZE*8+10,y))
    y+=30
    screen.blit(font_stats.render(f"Buffer: {rl_buffer_size}", True, (0,0,0)), (SQUARE_SIZE*8+10,y))
    y+=30
    if rl_pi:
        sorted_pi = sorted(rl_pi.items(), key=lambda x: -x[1])
        for move, prob in sorted_pi[:5]:
            screen.blit(font_stats.render(f"{move}: {prob:.2f}", True, (0,0,0)), (SQUARE_SIZE*8+10,y))
            y+=25

# --- AI/self-play function ---
def ai_move_mcts():
    """
    Изпълнява ход чрез MCTS, съхранява (state, policy_vector, value) в буфера,
    прави training step ако има >=32 примера и обновява rl_* променливите.
    """
    global last_move, rl_pi, rl_value, rl_loss, rl_buffer_size

    loss = 0.0
    pred_value_mean = 0.0
    pi_dict = {}

    if board.is_game_over():
        return {}, 0.0, 0.0

    # --- MCTS и избор на ход ---
    pi_dict = mcts_ai.run(board)           # dict: move -> prob
    move = mcts_ai.select_move(board)

    # --- Пешка промоция според policy ---
    piece = board.piece_at(move.from_square)
    if piece and piece.piece_type == chess.PAWN:
        if (board.turn == chess.WHITE and chess.square_rank(move.to_square) == 7) or \
           (board.turn == chess.BLACK and chess.square_rank(move.to_square) == 0):
            legal_promos = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
            promo_probs = np.array([pi_dict.get(chess.Move(move.from_square, move.to_square, promotion=p), 0.0) for p in legal_promos])
            if promo_probs.sum() > 0:
                promo_probs /= promo_probs.sum()
                move.promotion = np.random.choice(legal_promos, p=promo_probs)
            else:
                move.promotion = chess.QUEEN

    # --- Прилагане на хода ---
    board.push(move)
    last_move = (move.from_square, move.to_square)

    # --- Създаваме policy вектор (numpy) от pi_dict ---
    pi_vec = np.zeros(TOTAL_MOVES, dtype=np.float32)
    for mv, p in pi_dict.items():
        try:
            idx = move_to_index(mv)
            if 0 <= idx < TOTAL_MOVES:
                pi_vec[idx] = float(p)
        except Exception:
            continue

    # --- Състоянието (state) като numpy (15,8,8) ---
    tensor = fen_to_tensor(board.fen())
    if tensor.shape == (15, 8, 8):
        state_np = tensor.astype(np.float32)
    elif tensor.shape == (8, 8, 15):
        state_np = np.transpose(tensor, (2,0,1)).astype(np.float32)
    else:
        state_np = np.array(tensor, dtype=np.float32)
        if state_np.ndim == 3 and state_np.shape[-1] == 15:
            state_np = np.transpose(state_np, (2,0,1))

    # --- Запис в буфера ---
    with buffer_lock:
        buffer.push(state_np, pi_vec, 0.0)  # value=0.0 за текущото ходене
        rl_buffer_size = len(buffer)

    # --- Update value targets при края на играта ---
    if board.is_game_over():
        result_str = board.result()
        reward = 1.0 if result_str == "1-0" else -1.0 if result_str == "0-1" else 0.0
        with buffer_lock:
            for i in range(len(buffer.buffer)):
                s, p, _ = buffer.buffer[i]
                buffer.buffer[i] = (s, p, reward)
            rl_buffer_size = len(buffer)

    # --- Training step (ако има >=32) ---
    if len(buffer) >= 32:
        with buffer_lock:
            states_list, policies_list, values_list = buffer.sample(32)

        # stack and send to device
        states_batch = torch.stack(states_list).to(device)
        policies_batch = torch.stack(policies_list).to(device)
        values_batch = values_list.to(device)

        # forward + backward
        model.train()
        optimizer.zero_grad()
        logits, predicted_value = model(states_batch)
        target_classes = torch.argmax(policies_batch, dim=1)
        policy_loss = torch.nn.functional.cross_entropy(logits, target_classes)
        value_loss = torch.nn.functional.mse_loss(predicted_value, values_batch)
        total_loss = policy_loss + value_loss
        total_loss.backward()
        optimizer.step()

        loss = total_loss.item()
        pred_value_mean = predicted_value.mean().item()
    else:
        # forward pass само за текущото state за визуализация
        try:
            cur_state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(device)
            if cur_state.ndim == 4 and cur_state.shape[1] != 15:
                cur_state = cur_state.permute(0,3,1,2)
            with torch.no_grad():
                logits_c, pred_val_c = model(cur_state)
            probs_c = torch.softmax(logits_c, dim=1)[0].cpu().numpy()
            pred_value_mean = float(pred_val_c.mean().item())
            # proxy loss
            pi_t = torch.tensor(pi_vec, dtype=torch.float32).unsqueeze(0).to(device)
            tgt = torch.argmax(pi_t, dim=1).to(device)
            policy_loss_proxy = torch.nn.functional.cross_entropy(logits_c, tgt)
            value_loss_proxy = torch.nn.functional.mse_loss(pred_val_c, torch.tensor([[0.0]], dtype=torch.float32).to(device))
            loss = (policy_loss_proxy + value_loss_proxy).item()
        except Exception:
            pred_value_mean = 0.0
            loss = 0.0

    # --- Update RL live variables ---
    rl_pi = pi_dict
    rl_value = pred_value_mean
    rl_loss = loss
    rl_buffer_size = len(buffer)

    return pi_dict, rl_value, rl_loss

# --- Promotion popup ---
def ask_promotion_choice(screen):
    font_popup = pygame.font.SysFont("arial", 32, bold=True)
    options = [("Queen ♛", chess.QUEEN), ("Rook ♜", chess.ROOK), ("Bishop ♝", chess.BISHOP), ("Knight ♞", chess.KNIGHT)]
    screen_width, screen_height = screen.get_size()
    popup_width, popup_height = 280, 320
    popup_x = (screen_width-popup_width)//2
    popup_y = (screen_height-popup_height)//2
    popup_rect = pygame.Rect(popup_x, popup_y, popup_width, popup_height)
    pygame.draw.rect(screen, (255,192,203), popup_rect, border_radius=20)
    pygame.draw.rect(screen, (255,105,180), popup_rect, 4, border_radius=20)
    text = font_popup.render("Choose promotion:", True, (80,0,60))
    screen.blit(text, (popup_x+30, popup_y+20))
    buttons_popup=[]
    for i,(label,piece) in enumerate(options):
        rect = pygame.Rect(popup_x+50, popup_y+80+i*55, 180,45)
        pygame.draw.rect(screen, (255,105,180), rect, border_radius=12)
        txt = font_popup.render(label, True, (255,255,255))
        screen.blit(txt, (rect.x+15, rect.y+5))
        buttons_popup.append((rect,piece))
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type==pygame.MOUSEBUTTONDOWN:
                pos=pygame.mouse.get_pos()
                for rect,piece in buttons_popup:
                    if rect.collidepoint(pos):
                        return piece
            elif event.type==pygame.QUIT:
                pygame.quit()
                sys.exit()

# --- RL thread ---
def rl_self_play_loop():
    global board, move_number
    while True:
        if mode=="Self-Play" and not board.is_game_over():
            ai_move_mcts()
            move_number+=1
            time.sleep(0.05)
        else:
            time.sleep(0.1)

rl_thread = threading.Thread(target=rl_self_play_loop, daemon=True)
rl_thread.start()

# --- Main loop ---
pi=None
value=0.0
loss = 0.0

while True:
    screen.fill((255,255,255))
    draw_board(screen, board, selected_square)
    draw_status_bar()
    draw_buttons()
    draw_live_stats(screen, pi, value, move_number, loss)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            # --- Бутони ---
            for label, rect in buttons.items():
                if rect.collidepoint(pos):
                    if label=="New Game":
                        board = initial_board()
                        selected_square = None
                        last_move = None
                        move_number = 1
                    elif label=="Toggle Mode":
                        if mode=="Human vs Human":
                            mode="Human vs AI"
                        elif mode=="Human vs AI":
                            mode="Self-Play"
                        else:
                            mode="Human vs Human"
                    break
            else:
                # --- Дъска ---
                if pos[1]<SQUARE_SIZE*8 and not board.is_game_over():
                    square = get_square_from_mouse(pos)
                    if selected_square is None:
                        piece = board.piece_at(square)
                        if piece and piece.color==board.turn:
                            selected_square = square
                    else:
                        promotion = None
                        piece = board.piece_at(selected_square)
                        if piece and piece.piece_type==chess.PAWN:
                            if (board.turn==chess.WHITE and chess.square_rank(square)==7) or \
                               (board.turn==chess.BLACK and chess.square_rank(square)==0):
                                promotion = ask_promotion_choice(screen)
                        moved = make_move(board, selected_square, square, promotion)
                        if moved:
                            last_move = (selected_square, square)
                            selected_square = None
                            move_number += 1
                        else:
                            piece = board.piece_at(square)
                            if piece and piece.color==board.turn:
                                selected_square = square
                            else:
                                selected_square = None

    # --- AI ход (Human vs AI) ---
    if mode=="Human vs AI" and not board.is_game_over() and board.turn==chess.BLACK:
        pi, value, loss = ai_move_mcts()
        move_number += 1

    pygame.display.flip()
    clock.tick(30)
