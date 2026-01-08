import chess
from PySide6.QtCore import QObject, Signal


class AnalysisController(QObject):
    # Сигнал, че имаме нови данни от AI (за прерисуване на панела и heatmap)
    analysis_updated = Signal()

    # Сигнал за хайлайт на конкретен ход (при hover с мишката в списъка)
    highlight_move = Signal(object)

    def __init__(self):
        super().__init__()
        self.active = False
        self.current_value = 0.0
        self.top_moves = []  # Списък от (move_obj, san_str, prob, visits, pv_line)
        self.total_simulations = 0
        self.current_policy = {}  # За heatmap-а

        # НОВО: Запазваме ентропията (несигурността на AI)
        self.entropy = 0.0

    def update_analysis(self, info: dict):
        self.active = True
        self.current_value = info.get("value", 0.0)
        self.top_moves = info.get("top_moves", [])
        self.total_simulations = info.get("simulations", 0)
        self.current_policy = info.get("policy", {})

        # НОВО: Взимаме ентропията
        self.entropy = info.get("entropy", 0.0)

        self.analysis_updated.emit()

    def clear_analysis(self):
        """Изчиства данните при нов ход или рестарт."""
        self.active = False
        self.current_value = 0.0
        self.top_moves = []
        self.total_simulations = 0
        self.current_policy = {}
        self.entropy = 0.0  # НОВО

        self.analysis_updated.emit()
        self.set_hover_move(None)  # Махаме и хайлайта

    def set_hover_move(self, move: chess.Move | None):
        """Вика се от UI панела при движение на мишката."""
        self.highlight_move.emit(move)