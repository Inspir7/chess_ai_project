import chess
from PySide6.QtCore import QObject, Signal
from gui.game_mode import GameMode
from gui.players import HumanPlayer, AIPlayer


class GameController(QObject):
    board_changed = Signal()
    # Сигнал за грешка или статус (опционално)
    status_message = Signal(str)

    def __init__(self, mcts_engine=None, analysis_controller=None):
        super().__init__()
        self.board = chess.Board()
        self.mcts_engine = mcts_engine  # Твоята MCTS инстанция (ChessAI)
        self.analysis = analysis_controller  # Контролер за визуализация

        # НОВО: Време за изчакване между ходовете на AI (в милисекунди)
        # 50ms е "Fast", 1000ms е "Normal"
        self.ai_delay = 50

        # Текущи играчи
        self.white_player = HumanPlayer()
        self.black_player = HumanPlayer()

    # --- Mode Switching ---

    def set_mode(self, mode: GameMode):
        """Превключва режима и рестартира (или продължава) играта."""
        if mode == GameMode.HUMAN_VS_HUMAN:
            self.white_player = HumanPlayer()
            self.black_player = HumanPlayer()

        elif mode == GameMode.HUMAN_VS_AI:
            self.white_player = HumanPlayer()
            # Подаваме енджина на AI играча
            self.black_player = AIPlayer(self.mcts_engine)

        elif mode == GameMode.AI_VS_AI:
            self.white_player = AIPlayer(self.mcts_engine)
            self.black_player = AIPlayer(self.mcts_engine)

        # Можем да рестартираме играта при смяна на режим
        self.reset_game()

    def reset_game(self):
        self.board.reset()
        if self.analysis:
            self.analysis.clear_analysis()  # Чистим анализа при нова игра
        self.board_changed.emit()

    # --- Turn Logic ---

    def get_current_player(self):
        return self.white_player if self.board.turn == chess.WHITE else self.black_player

    def is_ai_turn(self) -> bool:
        """UI пита това, за да знае дали да чака AI."""
        if self.board.is_game_over():
            return False
        return self.get_current_player().is_ai()

    def play_ai_move(self):
        """Изпълнява хода на AI (ако е негов ред)."""
        player = self.get_current_player()

        if not player.is_ai():
            return

        move = None
        info = {}

        # Опитваме се да вземем подробна информация (Analysis)
        if self.mcts_engine and hasattr(self.mcts_engine, 'select_move_with_info'):
            move, info = self.mcts_engine.select_move_with_info(self.board.copy())
        else:
            # Fallback за съвместимост, ако методът липсва
            if player.is_ai():
                move = player.get_move(self.board.copy())

        if move:
            # Ако имаме анализ контролер, пращаме данните към него
            if self.analysis:
                self.analysis.update_analysis(info)

            self.push_move(move)

    def on_human_move(self, move: chess.Move):
        """Извиква се от UI при клик."""
        # Ако е ред на AI, игнорираме опитите на човека да мести
        if self.is_ai_turn():
            return

        # При ход на човек, старият анализ вече не е валиден
        if self.analysis:
            self.analysis.clear_analysis()

        self.push_move(move)

    def push_move(self, move: chess.Move):
        if move in self.board.legal_moves:
            self.board.push(move)
            self.board_changed.emit()