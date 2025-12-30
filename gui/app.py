import sys
import torch
import torch.nn as nn
import random
import math

from PySide6.QtGui import QColor, QFont, QPainter, QRadialGradient, QPixmap, QImage, QPen, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QRadioButton,
    QSlider,
    QLabel,
    QPushButton
)
from PySide6.QtCore import Qt, QRect, QTimer

from gui.board_widget import ChessBoardWidget, PieceRenderMode
from gui.game_controller import GameController
from gui.game_mode import GameMode

from gui.analysis_controller import AnalysisController
from gui.panels import TopMovesWidget

from training.mcts import MCTS
from training.move_encoding import get_total_move_count


# =========================================================
# DUMMY MODEL (За да тръгне играта веднага)
# =========================================================
class DummyChessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_size = get_total_move_count()

    def forward(self, x):
        batch_size = x.shape[0]
        policy_logits = torch.randn(batch_size, self.output_size)
        value = torch.tanh(torch.randn(batch_size, 1))
        return policy_logits, value


# =========================================================
# AI WRAPPER (Адаптер)
# =========================================================
class ChessAI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"AI using device: {self.device}")

        # Засега ползваме Dummy
        self.model = DummyChessModel().to(self.device)
        self.model.eval()

        self.mcts = MCTS(self.model, self.device, simulations=80)

    def select_move(self, board):
        move, _ = self.select_move_with_info(board)
        return move

    def select_move_with_info(self, board):
        try:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None, {}

            fake_policy = {}
            total_p = 0
            for m in legal_moves:
                p = random.random()
                fake_policy[m] = p
                total_p += p

            entropy = 0.0
            top_moves_data = []
            total_sims_demo = 80

            for m in legal_moves:
                prob = fake_policy[m] / total_p if total_p > 0 else 0
                if prob > 0:
                    entropy -= prob * math.log(prob)

                visits = int(prob * total_sims_demo)
                dummy_pv = f"{board.san(m)} ... (simulated line)"
                top_moves_data.append((m, board.san(m), prob, visits, dummy_pv))

            top_moves_data.sort(key=lambda x: x[2], reverse=True)
            best_move = legal_moves[0]
            fake_value = (random.random() * 2) - 1

            info = {
                "value": fake_value,
                "top_moves": top_moves_data[:6],
                "simulations": total_sims_demo,
                "policy": fake_policy,
                "entropy": entropy
            }

            print(f"AI избра: {best_move} (Eval: {fake_value:.2f}, Ent: {entropy:.2f})")
            return best_move, info

        except Exception as e:
            print(f"ГРЕШКА в MCTS: {e}")
            import traceback
            traceback.print_exc()
            return None, {}


# =========================================================
# MAIN WINDOW
# =========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Chess AI")
        self.setMinimumSize(1100, 650)

        # --- УМНО ЗАРЕЖДАНЕ НА ИКОНА (Code Fix) ---
        try:
            base_pixmap = QPixmap("assets/barbie/icon-logo.png")
            if not base_pixmap.isNull():
                square_size = 64
                square_pixmap = QPixmap(square_size, square_size)
                square_pixmap.fill(Qt.transparent)

                painter = QPainter(square_pixmap)
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setRenderHint(QPainter.SmoothPixmapTransform)

                scaled_logo = base_pixmap.scaled(square_size, square_size, Qt.KeepAspectRatio,
                                                 Qt.SmoothTransformation)
                x = (square_size - scaled_logo.width()) // 2
                y = (square_size - scaled_logo.height()) // 2

                painter.drawPixmap(x, y, scaled_logo)
                painter.end()

                self.setWindowIcon(QIcon(square_pixmap))
        except Exception as e:
            print(f"Could not load icon: {e}")

        self.analysis_controller = AnalysisController()
        ai_engine = ChessAI()

        self.controller = GameController(mcts_engine=ai_engine, analysis_controller=self.analysis_controller)
        self.controller.set_mode(GameMode.HUMAN_VS_HUMAN)

        # Свързваме сигнала за автоматично опресняване
        self.controller.board_changed.connect(self.update_ui)

        # Свързване на сигнала за промяна на имената на играчите
        if hasattr(self.controller, "players_changed"):
            self.controller.players_changed.connect(self.update_player_labels)

        central = QWidget()
        main_layout = QHBoxLayout(central)

        # --- КОНТЕЙНЕР ЗА ДЪСКАТА И ЕТИКЕТИТЕ ---
        central = QWidget()
        main_layout = QHBoxLayout(central)

        # Създаваме вертикален лейаут за дъската и имената под нея
        board_container_layout = QVBoxLayout()

        self.board = ChessBoardWidget(self.controller, self.analysis_controller)
        # stretch=1 тук казва на дъската да се разпъва максимално вертикално
        board_container_layout.addWidget(self.board, stretch=1)

        # Етикети за играчите
        self.player_white_lbl = QLabel("White: Human")
        self.player_black_lbl = QLabel("Black: Human")
        label_style = "font-weight: bold; color: #9E4F68; font-size: 13px; margin: 5px;"
        self.player_white_lbl.setStyleSheet(label_style)
        self.player_black_lbl.setStyleSheet(label_style)

        players_info_layout = QHBoxLayout()
        players_info_layout.addWidget(self.player_white_lbl)
        players_info_layout.addStretch()
        players_info_layout.addWidget(self.player_black_lbl)

        board_container_layout.addLayout(players_info_layout)

        # Добавяме контейнера на дъската в основния лейаут
        # stretch=1 тук позволява на дъската да заеме цялото ляво пространство
        main_layout.addLayout(board_container_layout, stretch=1)

        # --- ДЕСЕН ПАНЕЛ (Контроли и Анализ) ---
        right_panel = QVBoxLayout()
        controls = self._create_controls_panel()
        right_panel.addWidget(controls)

        self.analysis_widget = TopMovesWidget(self.analysis_controller)
        right_panel.addWidget(self.analysis_widget)
        right_panel.addStretch(1)

        right_container = QWidget()
        right_container.setLayout(right_panel)
        # ФИКСИРАМЕ ширината, за да не "яде" от мястото на дъската
        right_container.setFixedWidth(280)

        main_layout.addWidget(right_container)

        self.setCentralWidget(central)

        self.ai_thinking = False  # Флаг за предотвратяване на натрупване на таймери

        # Първоначално задаване на имената
        self.update_player_labels()

    def update_player_labels(self):
        """Обновява текста на етикетите според избрания режим и опонент."""
        white = self.controller.white_player
        black = self.controller.black_player

        def get_name(p):
            if isinstance(p, str):  # За ботовете от Benchmark режима (random, material, stockfish)
                return p.capitalize() + " Bot"
            if hasattr(p, "is_ai") and p.is_ai():
                return "Presie AI"
            return "Human"

        self.player_white_lbl.setText(f"White: {get_name(white)}")
        self.player_black_lbl.setText(f"Black: {get_name(black)}")

    def update_ui(self):
        """Предизвиква прерисуване на дъската и планира следващия AI ход."""
        self.board.update()

        # Проверяваме дали е ред на AI и дали вече не изчакваме ход, за да не се забързва
        if self.controller.is_ai_turn() and not self.ai_thinking:
            self.ai_thinking = True
            delay = max(10, self.controller.ai_delay)
            QTimer.singleShot(delay, self.run_ai_step)

    def run_ai_step(self):
        """Изпълнява AI хода и освобождава флага за следващия."""
        self.ai_thinking = False
        self.controller.play_ai_move()

    def _create_controls_panel(self) -> QWidget:
        panel = QGroupBox("Controls")

        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignTop)

        # --- БУТОН НОВА ИГРА (BARBIE STYLE) ---
        btn_new_game = QPushButton("New Game")
        btn_new_game.setCursor(Qt.PointingHandCursor)

        btn_new_game.setStyleSheet("""
            QPushButton {
                background-color: #FFE0E9;
                color: #9E4F68;
                border: 1px solid #F3C1D0;
                border-radius: 15px;
                padding: 8px;
                font-weight: bold;
                font-family: sans-serif;
            }
            QPushButton:hover {
                background-color: #FFB7C5;
                color: #6D2E40;
                border: 1px solid #E89CAD;
            }
            QPushButton:pressed {
                background-color: #E69AB0;
                color: white;
            }
        """)

        btn_new_game.clicked.connect(self.controller.reset_game)

        layout.addWidget(btn_new_game)
        layout.addSpacing(15)

        # --- Game Mode ---
        mode_group = QGroupBox("Game Mode")
        vbox_mode = QVBoxLayout()

        self.rb_pvp = QRadioButton("Human vs Human")
        self.rb_pve = QRadioButton("Human vs AI")
        self.rb_eve = QRadioButton("AI vs AI")

        self.rb_pvp.setChecked(True)

        self.rb_pvp.toggled.connect(lambda c: c and self.controller.set_mode(GameMode.HUMAN_VS_HUMAN))
        self.rb_pve.toggled.connect(lambda c: c and self.controller.set_mode(GameMode.HUMAN_VS_AI))
        self.rb_eve.toggled.connect(lambda c: c and self.controller.set_mode(GameMode.AI_VS_AI))

        vbox_mode.addWidget(self.rb_pvp)
        vbox_mode.addWidget(self.rb_pve)
        vbox_mode.addWidget(self.rb_eve)
        mode_group.setLayout(vbox_mode)

        layout.addWidget(mode_group)
        layout.addSpacing(10)

        # --- BENCHMARK MODE (AI vs Bots) ---
        bench_group = QGroupBox("Benchmark (AI vs Bots)")
        vbox_bench = QVBoxLayout()

        bench_style = """
            QPushButton {
                background-color: #F8F0F5;
                border: 1px solid #D8BFD8;
                border-radius: 10px;
                padding: 6px;
                color: #555;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #FFE4E1; }
        """

        btn_easy = QPushButton("vs Random Bot")
        btn_easy.setStyleSheet(bench_style)
        btn_easy.setCursor(Qt.PointingHandCursor)
        btn_easy.clicked.connect(lambda: self.controller.set_benchmark_mode("random"))
        vbox_bench.addWidget(btn_easy)

        btn_med = QPushButton("vs Material Bot")
        btn_med.setStyleSheet(bench_style)
        btn_med.setCursor(Qt.PointingHandCursor)
        btn_med.clicked.connect(lambda: self.controller.set_benchmark_mode("material"))
        vbox_bench.addWidget(btn_med)

        btn_sf = QPushButton("vs Stockfish")
        btn_sf.setStyleSheet(bench_style)
        btn_sf.setCursor(Qt.PointingHandCursor)
        btn_sf.clicked.connect(lambda: self.controller.set_benchmark_mode("stockfish"))
        vbox_bench.addWidget(btn_sf)

        bench_group.setLayout(vbox_bench)
        layout.addWidget(bench_group)
        layout.addSpacing(10)

        # --- AI Speed Control ---
        speed_group = QGroupBox("AI Speed")
        vbox_speed = QVBoxLayout()

        self.lbl_speed = QLabel("Fast (50ms)")
        self.lbl_speed.setAlignment(Qt.AlignCenter)
        self.lbl_speed.setStyleSheet("color: #666; font-size: 11px;")

        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(10, 2000)
        self.slider_speed.setValue(50)
        self.slider_speed.valueChanged.connect(self._on_speed_changed)

        vbox_speed.addWidget(self.lbl_speed)
        vbox_speed.addWidget(self.slider_speed)
        speed_group.setLayout(vbox_speed)

        layout.addWidget(speed_group)
        layout.addSpacing(10)

        # --- Display Options ---
        render_group = QGroupBox("Display Options")
        vbox_render = QVBoxLayout()

        self.radio_png = QRadioButton("Barbie pieces (SVG)")
        self.radio_unicode = QRadioButton("Unicode pieces")

        self.radio_png.setChecked(True)

        self.radio_png.toggled.connect(self._on_png_toggled)
        self.radio_unicode.toggled.connect(self._on_unicode_toggled)

        vbox_render.addWidget(self.radio_png)
        vbox_render.addWidget(self.radio_unicode)
        render_group.setLayout(vbox_render)

        layout.addWidget(render_group)
        return panel

    def _on_speed_changed(self, value):
        self.controller.ai_delay = value
        if value < 200:
            self.lbl_speed.setText(f"Fast ({value}ms)")
        elif value < 1000:
            self.lbl_speed.setText(f"Normal ({value}ms)")
        else:
            self.lbl_speed.setText(f"Slow ({value / 1000:.1f}s)")

    def _on_png_toggled(self, checked: bool):
        if checked:
            self.board.set_render_mode(PieceRenderMode.IMAGE)

    def _on_unicode_toggled(self, checked: bool):
        if checked:
            self.board.set_render_mode(PieceRenderMode.UNICODE)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()

    window.show()
    window.activateWindow()
    window.raise_()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()