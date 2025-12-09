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
# MCTS (AlphaZero-style)
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
        batch_size=16,
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

        self.batch_size = batch_size
        self.total_moves = get_total_move_count()

    # =====================================================
    # Public Interface
    # =====================================================
    def run(self, root_board, move_number=None):
        """
        Стартира MCTS търсене и връща dict {move: prob}.

        move_number: номер на полухода, за да контролираме
        Dirichlet шума в по-късните фази на партията.
        """
        root = MCTSNode(root_board.copy(stack=False))

        # Първоначална експанзия на root
        self._expand_node(root)

        # Добавяме Dirichlet noise ВЕДНЪЖ в началото
        self._apply_dirichlet_noise(root, move_number)

        pending = []

        for sim in range(self.simulations):
            node = root
            path = [node]

            # Selection
            while node.is_expanded and node.children:
                move, node = self._select_child(node)
                path.append(node)

            pending.append((node, path))

            # Batch-ова оценка
            if len(pending) >= self.batch_size or sim == self.simulations - 1:
                self._evaluate_batch(pending)
                pending = []

        return self._compute_root_policy(root)

    # =====================================================
    # Node expansion (single)
    # =====================================================
    def _expand_node(self, node):
        board = node.board

        if board.is_game_over():
            # value от гледна точка на ИГРАЧА НА ХОД
            res = board.result()
            if res == "1-0":
                value = 1.0 if board.turn == chess.WHITE else -1.0
            elif res == "0-1":
                value = 1.0 if board.turn == chess.BLACK else -1.0
            else:
                value = 0.0

            node.value_sum += value
            node.visit_count += 1
            node.is_expanded = True
            return

        legal = list(board.legal_moves)
        if not legal:
            node.value_sum += 0.0
            node.visit_count += 1
            node.is_expanded = True
            return

        # Encode state
        s = fen_to_tensor(board.fen())
        s = np.array(s, dtype=np.float32)

        # HWC → CHW
        if s.shape == (8, 8, 15):
            s = np.transpose(s, (2, 0, 1))
        elif s.shape != (15, 8, 8):
            s = s.reshape(15, 8, 8)

        state_tensor = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
            policy = torch.softmax(policy_logits, dim=1).squeeze().cpu().numpy()
            value = float(value.item())

        node.is_expanded = True

        # Деца
        for mv in legal:
            idx = move_to_index(mv)
            if idx is None or idx < 0 or idx >= self.total_moves:
                continue
            p = policy[idx]
            child_board = board.copy(stack=False)
            child_board.push(mv)
            node.children[mv] = MCTSNode(child_board, parent=node, prior=p)

        if not node.children:
            # safety fallback – равномерно
            mv = legal[0]
            child_board = board.copy(stack=False)
            child_board.push(mv)
            node.children[mv] = MCTSNode(child_board, parent=node, prior=1.0 / len(legal))

        return value

    # =====================================================
    # Dirichlet Noise at Root
    # =====================================================
    def _apply_dirichlet_noise(self, root, move_number):
        if not root.children:
            return

        # Noise само в ранната игра (примерно първите 30 полухода)
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
    # Batch Evaluation + Expansion
    # =====================================================
    def _evaluate_batch(self, pending):
        # split terminal and non-terminal
        terminal = []
        nonterm = []

        for node, path in pending:
            if node.board.is_game_over():
                terminal.append((node, path))
            else:
                nonterm.append((node, path))

        # Handle terminal nodes
        for node, path in terminal:
            res = node.board.result()
            board = node.board

            if res == "1-0":
                v = 1.0 if board.turn == chess.WHITE else -1.0
            elif res == "0-1":
                v = 1.0 if board.turn == chess.BLACK else -1.0
            else:
                v = 0.0

            self._backup(path, v)

        # Evaluate non-terminal nodes in batch
        if nonterm:
            boards = [n.board for (n, _) in nonterm]
            x = self._encode_boards(boards)

            with torch.no_grad():
                policy_logits, values = self.model(x)
                policy_logits = policy_logits.cpu().numpy()
                values = values.cpu().numpy().flatten()

            for i, (node, path) in enumerate(nonterm):
                logits = policy_logits[i]
                value = float(values[i])

                self._expand_with_logits(node, logits)
                self._backup(path, value)

    def _encode_boards(self, boards):
        """
        Бързо, безопасно batch-ване на бордове за MCTS inference.
        Работи на NumPy 2.0 без copy=False грешки.
        """
        np_arrs = []

        for b in boards:
            s = fen_to_tensor(b.fen())
            s = np.asarray(s, dtype=np.float32)

            # HWC → CHW
            if s.shape == (8, 8, 15):
                s = s.transpose(2, 0, 1)
            elif s.shape != (15, 8, 8):
                s = s.reshape(15, 8, 8)

            np_arrs.append(s)

        # Стекваме в масив (B, 15, 8, 8)
        batch_np = np.stack(np_arrs, axis=0).astype(np.float32)

        # Превръщаме в torch tensor по най-бързия начин
        return torch.from_numpy(batch_np).to(self.device)

    def _expand_with_logits(self, node, logits):
        policy = torch.softmax(torch.tensor(logits), dim=0).numpy()

        legal = list(node.board.legal_moves)
        node.is_expanded = True

        for mv in legal:
            idx = move_to_index(mv)
            if idx is None or idx < 0 or idx >= self.total_moves:
                continue
            p = policy[idx]
            newb = node.board.copy(stack=False)
            newb.push(mv)
            node.children[mv] = MCTSNode(newb, parent=node, prior=float(p))

    # =====================================================
    # Backup
    # =====================================================
    def _backup(self, path, value):
        v = value
        for node in reversed(path):
            node.value_sum += v
            node.visit_count += 1
            v = -v  # flip perspective

    # =====================================================
    # Final Policy at Root
    # =====================================================
    def _compute_root_policy(self, root):
        moves = list(root.children.keys())
        visits = np.array([root.children[mv].visit_count for mv in moves], dtype=np.float32)

        if visits.sum() == 0:
            probs = np.ones(len(moves), dtype=np.float32) / len(moves)
        else:
            probs = visits / visits.sum()

        return {mv: float(p) for mv, p in zip(moves, probs)}