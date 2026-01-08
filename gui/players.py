import chess
from abc import ABC, abstractmethod

class Player(ABC):
    @abstractmethod
    def is_ai(self) -> bool:
        """Връща True, ако играчът е компютър и изисква автоматично извикване."""
        pass

    @abstractmethod
    def get_move(self, board: chess.Board) -> chess.Move | None:
        """Изчислява ход. За Human връща None (ходът идва от UI)."""
        pass


class HumanPlayer(Player):
    def is_ai(self) -> bool:
        return False

    def get_move(self, board: chess.Board) -> chess.Move | None:
        return None  # Ходът се чака от кликване в UI


class AIPlayer(Player):
    def __init__(self, mcts_engine):
        self.mcts = mcts_engine

    def is_ai(self) -> bool:
        return True

    def get_move(self, board: chess.Board) -> chess.Move | None:
        # Тук викаме твоя MCTS алгоритъм
        if board.is_game_over():
            return None
        return self.mcts.select_move(board)