# ne e nujno
import chess
import chess.svg
import matplotlib.pyplot as plt
from cairosvg import svg2png
from io import BytesIO
from PIL import Image

def show_fen(fen):
    board = chess.Board(fen)

    # Конвертираме шахматната дъска в SVG
    svg_data = chess.svg.board(board=board)

    # Конвертираме SVG в PNG с помощта на cairosvg
    png_data = svg2png(bytestring=svg_data)

    # Зареждаме PNG в PIL и го показваме с matplotlib
    image = Image.open(BytesIO(png_data))
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis("off")  # Премахваме осите
    plt.show()

# Тествай с началната шахматна позиция
start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
show_fen(start_fen)
