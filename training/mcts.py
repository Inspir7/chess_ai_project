import math
import numpy as np
import torch
import torch.nn.functional as F
import chess

from data.generate_labeled_data import fen_to_tensor
from training.move_encoding import move_to_index, get_total_move_count, flip_move_index


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
        simulations=160,
        c_puct=3.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        batch_size=16,  # запазваме аргумента, макар да не се ползва
    ):
        self.model = model
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct

        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        self.total_moves = get_total_move_count()
        self.last_root = None

    # =====================================================
    # Public Interface
    # =====================================================
    def run(self, root_board, move_number=None):
        """
        Стартира MCTS търсене и връща dict {move: prob}.
        """
        root = MCTSNode(root_board.copy(stack=False))
        self.last_root = root

        # 1. Expand Root
        _ = self._evaluate_and_expand(root)

        # 2. Add Noise
        self._apply_dirichlet_noise(root, move_number)

        # 3. Main Loop
        for _ in range(self.simulations):
            node = root
            path = [node]

            # Select
            while node.is_expanded and node.children:
                move, node = self._select_child(node)
                path.append(node)

            # Expand & Eval
            value = self._evaluate_and_expand(node)

            # Backup
            self._backup(path, value)

        return self._compute_root_policy(root)

    # =====================================================
    # Evaluate + Expand (CRITICAL FIXES HERE)
    # =====================================================
    def _evaluate_and_expand(self, node):
        board = node.board

        # Проверка за край на играта
        if board.is_game_over():
            result = board.result()
            # Връщаме резултата от гледна точка на текущия играч на хода
            if result == "1-0": return 1.0 if board.turn == chess.WHITE else -1.0
            if result == "0-1": return 1.0 if board.turn == chess.BLACK else -1.0
            return 0.0

        # 1. INPUT
        state_tensor = fen_to_tensor(board)
        state_tensor = torch.tensor(state_tensor, dtype=torch.float32).unsqueeze(0)
        state_tensor = state_tensor.permute(0, 3, 1, 2).to(self.device)

        # 2. INFERENCE
        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)

        leaf_value = value.item()
        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

        # 3. OUTPUT FLIP (FIXED)
        if board.turn == chess.BLACK:
            fixed_policy = np.zeros_like(policy_probs)
            for i in range(len(policy_probs)):
                if policy_probs[i] > 1e-8: # лека оптимизация
                    mirror_idx = flip_move_index(i)
                    if mirror_idx < len(fixed_policy):
                        fixed_policy[mirror_idx] = policy_probs[i]
            policy_probs = fixed_policy

        # 4. EXPAND (FIXED CONSTRUCTOR)
        legal_moves = list(board.legal_moves)
        policy_sum = 0
        node.is_expanded = True

        for move in legal_moves:
            idx = move_to_index(move)
            if idx is not None and 0 <= idx < len(policy_probs):
                prior = policy_probs[idx]

                if move not in node.children:
                    # FIX: Създаваме нова дъска и я подаваме на конструктора
                    next_board = board.copy()
                    next_board.push(move)
                    node.children[move] = MCTSNode(next_board, parent=node, prior=prior)

                policy_sum += prior

        # Renormalize
        if policy_sum > 0:
            for child in node.children.values():
                child.prior /= policy_sum
        else:
            for child in node.children.values():
                child.prior = 1.0 / len(legal_moves)

        return leaf_value

    # =====================================================
    # Helpers
    # =====================================================
    def _apply_dirichlet_noise(self, root, move_number):
        if not root.children: return
        if move_number is not None and move_number > 30: return

        eps = self.dirichlet_epsilon
        moves = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))

        for mv, n in zip(moves, noise):
            child = root.children[mv]
            child.prior = (1.0 - eps) * child.prior + eps * float(n)

    def _select_child(self, node):
        best_score = -float('inf')
        best_move = None
        best_child = None

        parent_visits = max(1, node.visit_count)
        sqrt_parent = math.sqrt(parent_visits)

        for mv, child in node.children.items():
            # FIX: Използваме -q, защото child.q_value е добър за опонента
            q = -child.q_value
            u = self.c_puct * child.prior * (sqrt_parent / (1 + child.visit_count))
            score = q + u

            if score > best_score:
                best_score = score
                best_move = mv
                best_child = child

        return best_move, best_child

    def _backup(self, path, value):
        v = value
        for node in reversed(path):
            node.value_sum += v
            node.visit_count += 1
            v = -v

    def _compute_root_policy(self, root):
        moves = list(root.children.keys())
        visits = np.array([root.children[mv].visit_count for mv in moves], dtype=np.float32)

        if visits.sum() == 0:
            probs = np.ones(len(moves), dtype=np.float32) / len(moves)
        else:
            probs = visits / visits.sum()

        return {mv: float(p) for mv, p in zip(moves, probs)}

    def run_with_Q(self, board, move_number=None):
        pi = self.run(board, move_number)
        root = self.last_root
        Q = {}
        if root is not None and root.children:
            for move, child in root.children.items():
                Q[move] = float(child.q_value)
        else:
            for move in pi.keys():
                Q[move] = 0.0
        return pi, Q