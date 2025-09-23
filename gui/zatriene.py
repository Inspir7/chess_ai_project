import pygame
import chess

pygame.init()

WIDTH, HEIGHT = 640, 640
SQUARE_SIZE = WIDTH // 8
WHITE = (255, 255, 255)
LIGHT_PINK = (255, 182, 193)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Test Chess GUI")
board = chess.Board()

piece_images = {
        'P': pygame.image.load("assets/w_P.png"),
        'R': pygame.image.load("assets/w_R.png"),
        'N': pygame.image.load("assets/w_N.png"),
        'B': pygame.image.load("assets/w_B.png"),
        'Q': pygame.image.load("assets/w_Q.png"),
        'K': pygame.image.load("assets/w_K.png"),
        'p': pygame.image.load("assets/b_P.png"),
        'r': pygame.image.load("assets/b_R.png"),
        'n': pygame.image.load("assets/b_N.png"),
        'b': pygame.image.load("assets/b_B.png"),
        'q': pygame.image.load("assets/b_Q.png"),
        'k': pygame.image.load("assets/b_K.png"),
    }

def draw_board():
    colors = [pygame.Color("white"), pygame.Color("lightpink")]
    for rank in range(8):
        for file in range(8):
            color = colors[(rank + file) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(file*SQUARE_SIZE, rank*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_pieces():
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_str = piece.symbol()
            file = chess.square_file(square)
            rank = 7 - chess.square_rank(square)
            image = piece_images[piece_str]
            screen.blit(pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE)),
                        (file * SQUARE_SIZE, rank * SQUARE_SIZE))

def main():
    global mode, board, replay_buffer
    selected_square = None
    running = True

    while running:
        draw_board()
        draw_pieces()
        btn_rect, toggle_btn_rect = draw_info_panel()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_replay_buffer(replay_buffer, "C:\\Users\\prezi\\PycharmProjects\\chess_ai_project\\utils\\replays\\replay_buffer.pkl")
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Проверка за бутони
                if btn_rect.collidepoint(event.pos):
                    board.reset()
                    replay_buffer = []
                    selected_square = None
                    continue
                elif toggle_btn_rect.collidepoint(event.pos):
                    mode = "self_play" if mode == "human_vs_ai" else "human_vs_ai"
                    print(f"Mode changed to: {mode}")
                    continue

                # Проверка за дъска
                file = event.pos[0] // SQUARE_SIZE
                rank = 7 - (event.pos[1] // SQUARE_SIZE)
                square = chess.square(file, rank)

                if selected_square is None:
                    if board.piece_at(square) and board.piece_at(square).color == board.turn:
                        selected_square = square
                        print(f"Selected: {chess.square_name(square)}")
                else:
                    if square != selected_square:
                        move = chess.Move(selected_square, square)

                        # Ако е промоция
                        if (board.piece_at(selected_square).piece_type == chess.PAWN and
                            chess.square_rank(square) in [0, 7]):
                            promotion_piece = show_promotion_menu(screen, font)
                            move = chess.Move(selected_square, square, promotion=promotion_piece)

                        if move in board.legal_moves:
                            board.push(move)
                            replay_buffer.append((board.fen(), None))
                            print(f"Move played: {move.uci()}")

                            selected_square = None

                            # AI ход
                            if mode == "human_vs_ai" and not board.is_game_over():
                                ai_move = ai_agent.select_move(board)
                                board.push(ai_move)
                                replay_buffer.append((board.fen(), None))
                                print(f"AI move: {ai_move.uci()}")
                        else:
                            print(f"Illegal move: {move.uci()}")
                            selected_square = None
                    else:
                        selected_square = None


if __name__ == "__main__":
    main()
