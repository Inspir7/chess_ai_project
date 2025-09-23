# engine/promotion.py
import pygame
import chess

def show_promotion_menu(screen, board, square, is_white):
    font = pygame.font.SysFont("Arial", 24)
    options = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    rects = []

    menu_width = 120
    menu_height = 30 * len(options)
    x, y = (square % 8) * 80, (7 - square // 8) * 80
    menu_rect = pygame.Rect(x, y - menu_height if y > 200 else y + 80, menu_width, menu_height)

    for i, piece in enumerate(options):
        rect = pygame.Rect(menu_rect.x, menu_rect.y + i * 30, menu_width, 30)
        rects.append((rect, piece))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                for rect, piece in rects:
                    if rect.collidepoint(mx, my):
                        return piece
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        pygame.draw.rect(screen, (220, 220, 220), menu_rect)
        for i, (rect, piece) in enumerate(rects):
            label = font.render(chess.Piece(piece, is_white).symbol(), True, (0, 0, 0))
            screen.blit(label, (rect.x + 10, rect.y + 5))
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)

        pygame.display.flip()
