import math
import numpy as np
import torch
import torch.nn.functional as F
import chess

from data.generate_labeled_data import fen_to_tensor
from training.move_encoding import move_to_index, get_total_move_count


# =====================================================
# MCTS Node
# =====================================================

class MCTSNode:
    __slots__ = ("board", "parent", "prior", "visit_count", "value_sum", "children", "is_expanded")

    def __init__(self, board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.prior = float(prior)

        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

        self.is_expanded = False

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


# =====================================================
# MCTS (AlphaZero-style, без batch усложнения)
# =====================================================

class MCTS:
    def __init__(
        self,
        model,
        device,
        simulations=200,
        c_puct=1.2,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        batch_size=16,  # вече не се ползва, оставям го за съвместимост
    ):
        """
        model: AlphaZero policy-value network
        device: torch.device
        simulations: брой MCTS симулации за ход
        c_puct: коефициент за баланс exploration / exploitation
        dirichlet_*: параметри за root noise (само на корена)
        """
        self.model = model
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct

        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        self.total_moves = get_total_move_count()

    # =====================================================
    # Public Interface
    # =====================================================
    def run(self, root_board, move_number=None):
        """
        Стартира MCTS търсене и връща dict {move: prob} (π по visit count-ове).
        """
        root = MCTSNode(root_board.copy(stack=False))

        # Първоначална експанзия на root (NN + деца)
        value_root = self._evaluate_and_expand(root)

        # добавяме Dirichlet noise веднъж в началото (opening)
        self._apply_dirichlet_noise(root, move_number)

        # Основен loop: една симулация = selection → expansion → backup
        for _ in range(self.simulations):
            node = root
            path = [node]

            # Selection: слизаме по дървото според PUCT,
            # докато стигнем неекспандиран node или терминален
            while node.is_expanded and node.children:
                move, node = self._select_child(node)
                path.append(node)

            # Expansion + NN оценка
            value = self._evaluate_and_expand(node)

            # Backup (flip на знака по пътя)
            self._backup(path, value)

        return self._compute_root_policy(root)

    # =====================================================
    # Evaluate + Expand (single node)
    # =====================================================
    def _evaluate_and_expand(self, node):
        """
        Оценява node с мрежата и го експандира (ако не е терминален).
        Връща value от гледна точка на ИГРАЧА НА ХОД в този node.
        """
        board = node.board

        # --------- Терминален случай ---------
        if board.is_game_over():
            res = board.result()
            # value винаги от гледна точка на играча на ход
            if res == "1-0":      # бели са победили
                value = 1.0 if board.turn == chess.WHITE else -1.0
            elif res == "0-1":    # черни са победили
                value = 1.0 if board.turn == chess.BLACK else -1.0
            else:
                value = 0.0

            node.is_expanded = True  # няма деца
            return value

        legal = list(board.legal_moves)
        if not legal:
            # няма легални ходове, но не е маркирано като game_over (рядко)
            node.is_expanded = True
            return 0.0

        # --------- Encode state ---------
        s = fen_to_tensor(board.fen())
        s = np.asarray(s, dtype=np.float32)

        # HWC → CHW
        if s.shape == (8, 8, 15):
            s = s.transpose(2, 0, 1)
        elif s.shape != (15, 8, 8):
            s = s.reshape(15, 8, 8)

        state_tensor = torch.from_numpy(s).unsqueeze(0).to(self.device)

        # --------- Neural net inference ---------
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
            # value: скалар в [-1,1], от гледна точка на ИГРАЧА НА ХОД (както сме тренирали)
            value = float(value.item())
            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

        # --------- Експанзия: деца с prior-и ---------
        node.is_expanded = True
        node.children = {}

        for mv in legal:
            idx = move_to_index(mv)
            if idx is None or idx < 0 or idx >= self.total_moves:
                continue
            p = float(policy[idx])
            child_board = board.copy(stack=False)
            child_board.push(mv)
            node.children[mv] = MCTSNode(child_board, parent=node, prior=p)

        # safety fallback – ако по някаква причина всичко е skip-нато
        if not node.children:
            uniform_p = 1.0 / len(legal)
            for mv in legal:
                child_board = board.copy(stack=False)
                child_board.push(mv)
                node.children[mv] = MCTSNode(child_board, parent=node, prior=uniform_p)

        return value

    # =====================================================
    # Dirichlet Noise at Root
    # =====================================================
    def _apply_dirichlet_noise(self, root, move_number):
        """
        Добавяме Dirichlet noise към prior-ите на root децата.
        Ползваме го основно в opening (примерно до 30-ти полуход).
        """
        if not root.children:
            return

        if move_number is not None and move_number > 30:
            return

        eps = self.dirichlet_epsilon
        moves = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))

        for mv, n in zip(moves, noise):
            child = root.children[mv]
            child.prior = (1.0 - eps) * child.prior + eps * float(n)

    # =====================================================
    # Selection (PUCT)
    # =====================================================
    def _select_child(self, node):
        """
        Избираме дете по стандартната PUCT формула:
        score = Q + U, където U ~ prior * sqrt(N) / (1 + n)
        """
        best_score = -1e9
        best_move = None
        best_child = None

        parent_visits = max(1, node.visit_count)
        sqrt_parent = math.sqrt(parent_visits)

        for mv, child in node.children.items():
            q = child.q_value
            u = self.c_puct * child.prior * (sqrt_parent / (1 + child.visit_count))
            score = q + u
            if score > best_score:
                best_score = score
                best_move = mv
                best_child = child

        return best_move, best_child

    # =====================================================
    # Backup
    # =====================================================
    def _backup(self, path, value):
        """
        Backprop на value по пътя root → leaf, като flip-ваме знака,
        защото value е винаги от гледна точка на играча на ход.
        """
        v = value
        for node in reversed(path):
            node.value_sum += v
            node.visit_count += 1
            v = -v  # flip perspective при смяна на ход

    # =====================================================
    # Final Policy at Root
    # =====================================================
    def _compute_root_policy(self, root):
        """
        Връща π като {move: prob} на база visit counts от root-а.
        """
        moves = list(root.children.keys())
        visits = np.array([root.children[mv].visit_count for mv in moves], dtype=np.float32)

        if visits.sum() == 0:
            probs = np.ones(len(moves), dtype=np.float32) / len(moves)
        else:
            probs = visits / visits.sum()

        return {mv: float(p) for mv, p in zip(moves, probs)}