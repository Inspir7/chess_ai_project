import chess
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QProgressBar, QHBoxLayout)
from PySide6.QtCore import Qt
from gui.analysis_controller import AnalysisController


class EvaluationBar(QProgressBar):
    def __init__(self):
        super().__init__()
        self.setOrientation(Qt.Horizontal)
        self.setRange(0, 200)  # 0 = Черни печелят, 200 = Бели печелят
        self.setValue(100)  # 100 = Равно
        self.setTextVisible(False)
        self.setStyleSheet("""
            QProgressBar {
                border: 1px solid #D0D0D0;
                border-radius: 4px;
                background-color: #505050; /* Тъмна страна */
                min-height: 12px;
                max-height: 12px;
            }
            QProgressBar::chunk {
                background-color: #FF99AA; /* Светла страна (Розово) */
                border-radius: 3px;
            }
        """)

    def update_eval(self, value):
        # value е от -1.0 до 1.0 -> мапваме към 0..200
        int_val = int((value + 1.0) * 100)
        self.setValue(int_val)


class TopMovesWidget(QWidget):
    def __init__(self, analysis_controller: AnalysisController):
        super().__init__()
        self.analysis = analysis_controller
        self.analysis.analysis_updated.connect(self._refresh)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 0)

        # --- HEADER (Sims + Entropy) ---
        header_layout = QHBoxLayout()

        self.title_lbl = QLabel("AI Analysis")
        self.title_lbl.setStyleSheet("font-weight: bold; color: #444;")

        # НОВО: Етикет за несигурност
        self.entropy_lbl = QLabel("Uncertainty: --")
        self.entropy_lbl.setStyleSheet("font-size: 10px; color: #888;")
        self.entropy_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        header_layout.addWidget(self.title_lbl)
        header_layout.addWidget(self.entropy_lbl)
        layout.addLayout(header_layout)

        # Бар за оценка
        self.eval_bar = EvaluationBar()
        layout.addWidget(self.eval_bar)

        # Списък с ходове
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #FFF0F5; /* Lavender Blush */
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                color: #333;
                font-family: monospace; /* Моноширинен шрифт за подравняване */
            }
            QListWidget::item {
                padding: 4px;
            }
            QListWidget::item:hover {
                background-color: #FFD1DC; /* Лек highlight на реда */
            }
        """)

        # Включваме проследяване на мишката за Hover ефекта върху дъската
        self.list_widget.setMouseTracking(True)
        self.list_widget.itemEntered.connect(self._on_item_hover)

        # Когато мишката излезе от списъка, махаме хайлайта от дъската
        self.list_widget.viewport().leaveEvent = lambda e: self.analysis.set_hover_move(None)

        layout.addWidget(self.list_widget)

    def _refresh(self):
        # 1. Header Info
        sims = self.analysis.total_simulations
        ent = self.analysis.entropy

        # Определяме текст според ентропията
        if ent < 0.6:
            ent_str = "Low"
        elif ent < 1.3:
            ent_str = "Med"
        else:
            ent_str = "High"

        self.title_lbl.setText(f"Depth: {sims} sims")
        self.entropy_lbl.setText(f"Uncertainty: {ent_str} ({ent:.2f})")

        # 2. Eval
        val = self.analysis.current_value
        self.eval_bar.update_eval(val)
        self.eval_bar.setToolTip(f"Eval: {val:.2f} (White perspective)")

        # 3. List
        self.list_widget.clear()
        moves = self.analysis.top_moves

        if not self.analysis.active:
            self.list_widget.addItem("Waiting for AI...")
            return

        for data in moves:
            # Разопаковане (с fallback, ако няма PV, за безопасност)
            if len(data) == 5:
                move_obj, san, prob, visits, pv_line = data
            else:
                move_obj, san, prob, visits = data
                pv_line = "N/A"

            # Форматираме текста: "e4     (142) 45%"
            text = f"{san:<6} ({visits:>3}) {prob * 100:.0f}%"

            item = QListWidgetItem(text)
            # ВАЖНО: Запазваме обекта на хода, за да го ползваме при hover
            item.setData(Qt.UserRole, move_obj)

            # НОВО: Tooltip с прогнозираната линия!
            item.setToolTip(f"<b>Line:</b> {pv_line}<br><i>Probability: {prob:.4f}</i>")

            self.list_widget.addItem(item)

    def _on_item_hover(self, item):
        """Вика се, когато мишката мине върху ред."""
        move = item.data(Qt.UserRole)
        self.analysis.set_hover_move(move)