import math
import torch
import torch.nn.functional as F
import chess
import numpy as np
from training.generate_labeled_data import fen_to_tensor
from models.move_encoding import move_to_index, index_to_move

class MCTSNode:
    def __init__(self, board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

class MCTS:
    def __init__(self, model, device, simulations=800, c_puct=1.0, temperature=1.0, verbose=False):
        self.model = model
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.verbose = verbose

    def run(self, root_board):
        root = MCTSNode(root_board.copy())

        if self.verbose:
            print("[MCTS] Starting search...")

        # Initial expansion
        policy, _ = self._model_eval(root.board)
        for move, p in policy.items():
            next_board = root.board.copy()
            next_board.push(move)
            root.children[move] = MCTSNode(next_board, parent=root, prior=p)

        for sim in range(self.simulations):
            node = root
            search_path = [node]

            # Selection
            while node.expanded():
                move, node = self._select_child(node)
                search_path.append(node)

            # Expansion
            if not node.board.is_game_over():
                policy, value = self._model_eval(node.board)
                for mv, p in policy.items():
                    if mv not in node.children:
                        child_board = node.board.copy()
                        child_board.push(mv)
                        node.children[mv] = MCTSNode(child_board, parent=node, prior=p)
            else:
                result = node.board.result(claim_draw=True)
                value = 1.0 if result == '1-0' else -1.0 if result == '0-1' else 0.0

            # Backpropagation
            for nd in reversed(search_path):
                nd.visit_count += 1
                nd.value_sum += value
                value = -value

            if self.verbose and (sim + 1) % 50 == 0:
                print(f"[MCTS] Simulation {sim + 1}/{self.simulations} complete")

        visits = {move: child.visit_count for move, child in root.children.items()}
        pi = self._apply_temperature(visits)

        if self.verbose:
            print("[MCTS] Final move probabilities:")
            for mv, prob in sorted(pi.items(), key=lambda x: -x[1]):
                print(f"  {mv}: {round(prob, 3)}")

        return pi

    def _select_child(self, node):
        best_score = -float('inf')
        best_move, best_child = None, None

        for move, child in node.children.items():
            u = self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
            score = child.value() + u
            if score > best_score:
                best_score, best_move, best_child = score, move, child

        return best_move, best_child

    def _safe_scalar(self, prob):
        if isinstance(prob, np.ndarray):
            prob = prob.squeeze()
            if prob.size != 1:
                raise ValueError(f"Expected scalar, got array with shape {prob.shape}")
            return prob.item()
        return float(prob)

    def _model_eval(self, board):
        np_tensor = fen_to_tensor(board.fen())
        x = torch.tensor(np_tensor, dtype=torch.float32, device=self.device).unsqueeze(0).permute(0, 3, 1, 2)

        with torch.no_grad():
            logits, value = self.model(x)

        probs = F.softmax(logits, dim=1)[0].cpu()
        legal = list(board.legal_moves)
        policy = {}

        for move in legal:
            try:
                idx = move_to_index(move)
                if idx == -1 or idx >= len(probs):
                    continue
                prob = self._safe_scalar(probs[idx])
                policy[move] = prob
            except Exception as e:
                if self.verbose:
                    print(f"[MCTS] Warning: Skipping move {move} due to error: {e}")

        total = sum(policy.values())
        if total > 0.0:
            for mv in policy:
                policy[mv] /= total
        else:
            uniform = 1.0 / len(legal)
            policy = {mv: uniform for mv in legal}

        return policy, value.item()

    def _apply_temperature(self, visits):
        temp = self.temperature
        if temp == 0:
            best_move = max(visits.items(), key=lambda x: x[1])[0]
            return {move: 1.0 if move == best_move else 0.0 for move in visits}

        visits_temp = {move: count ** (1 / temp) for move, count in visits.items()}
        total = sum(visits_temp.values())
        return {move: prob / total for move, prob in visits_temp.items()}

    def select_move(self, board):
        pi = self.run(board)
        best_move = max(pi.items(), key=lambda x: x[1])[0]
        return best_move
