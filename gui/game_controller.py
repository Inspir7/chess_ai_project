import chess
import chess.engine
import random
from PySide6.QtCore import QObject, Signal
from gui.game_mode import GameMode
from gui.players import HumanPlayer, AIPlayer


class GameController(QObject):
    board_changed = Signal()
    # НОВО: Сигнал за обновяване на етикетите с имената на играчите в UI
    players_changed = Signal()

    def __init__(self, mcts_engine=None, analysis_controller=None):
        super().__init__()
        self.board = chess.Board()
        self.mcts_engine = mcts_engine
        self.analysis = analysis_controller
        self.ai_delay = 50
        self.white_player = HumanPlayer()
        self.black_player = HumanPlayer()
        self.stockfish_path = "/usr/games/stockfish"

    def set_mode(self, mode):
        if mode == GameMode.HUMAN_VS_HUMAN:
            self.white_player, self.black_player = HumanPlayer(), HumanPlayer()
        elif mode == GameMode.HUMAN_VS_AI:
            self.white_player, self.black_player = HumanPlayer(), AIPlayer(self.mcts_engine)
        elif mode == GameMode.AI_VS_AI:
            self.white_player, self.black_player = AIPlayer(self.mcts_engine), AIPlayer(self.mcts_engine)

        # НОВО: Излъчваме сигнал, че играчите са се променили
        self.players_changed.emit()
        self.reset_game()

    def set_benchmark_mode(self, bot_type):
        """Твоят AI винаги е Бели срещу избран Бот (Черни)."""
        self.white_player = AIPlayer(self.mcts_engine)
        self.black_player = bot_type  # "random", "material", "stockfish"

        # НОВО: Излъчваме сигнал, че играчите са се променили
        self.players_changed.emit()
        self.reset_game()

    def is_ai_turn(self):
        if self.board.is_game_over(): return False
        p = self.white_player if self.board.turn == chess.WHITE else self.black_player
        return isinstance(p, (AIPlayer, str))

    def play_ai_move(self):
        if self.board.is_game_over(): return
        p = self.white_player if self.board.turn == chess.WHITE else self.black_player

        move = None
        # AlphaZero
        if isinstance(p, AIPlayer):
            move, info = self.mcts_engine.select_move_with_info(self.board.copy())
            if self.analysis: self.analysis.update_analysis(info)
        # Ботове
        elif p == "random":
            move = random.choice(list(self.board.legal_moves))
        elif p == "material":
            move = self._get_material_move()
        elif p == "stockfish":
            move = self._get_stockfish_move()

        if move:
            self.board.push(move)
            self.board_changed.emit()

    def _get_material_move(self):
        vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        moves = list(self.board.legal_moves)
        best_score = -9999
        best_moves = [moves[0]]
        for m in moves:
            self.board.push(m)
            # Оценяваме спрямо играча, чийто ред е бил преди хода
            score = sum(len(self.board.pieces(pt, not self.board.turn)) * v for pt, v in vals.items())
            self.board.pop()
            if score > best_score:
                best_score, best_moves = score, [m]
            elif score == best_score:
                best_moves.append(m)
        return random.choice(best_moves)

    def _get_stockfish_move(self):
        try:
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as eng:
                res = eng.play(self.board, chess.engine.Limit(time=0.05))
                return res.move
        except:
            return random.choice(list(self.board.legal_moves))

    def on_human_move(self, move):
        if self.board.is_game_over() or self.is_ai_turn(): return
        self.board.push(move)
        if self.analysis: self.analysis.clear_analysis()
        self.board_changed.emit()

    def reset_game(self):
        self.board.reset()
        if self.analysis: self.analysis.clear_analysis()
        self.board_changed.emit()