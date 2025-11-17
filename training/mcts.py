import math
import torch
import torch.nn.functional as F
import numpy as np
from data.generate_labeled_data import fen_to_tensor
from models.move_encoding import move_to_index


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
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


class MCTS:
    def __init__(self, model, device, simulations=800, c_puct=1.0, temperature=1.0, verbose=False):
        self.model = model
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.verbose = verbose
        self.root = None

    # --- utility to perform a cheaper terminal check first ---
    def _is_terminal(self, board):
        """
        Cheap terminal checks first (very fast): checkmate, stalemate, insufficient material.
        Only if those are False we fall back to a full is_game_over check with claim_draw=False
        to avoid expensive repetition scans in the common case.
        """
        try:
            if board.is_checkmate():
                return True, 1.0 if board.turn == False else -1.0  # if checkmate, previous player won
            if board.is_stalemate():
                return True, 0.0
            if board.is_insufficient_material():
                return True, 0.0
        except Exception:
            # if any board method errors (shouldn't), fallback to full check below
            pass

        # fallback: do a full game-over check but avoid expensive claim_draw by default
        try:
            game_over = board.is_game_over(claim_draw=False)
        except TypeError:
            # older python-chess: no claim_draw param
            game_over = board.is_game_over()
        if game_over:
            try:
                res = board.result(claim_draw=False)
            except Exception:
                try:
                    res = board.result()
                except Exception:
                    res = "1/2-1/2"
            if res == "1-0":
                return True, 1.0
            elif res == "0-1":
                return True, -1.0
            else:
                return True, 0.0
        return False, 0.0

    def run(self, root_board):
        # Use stack=False to avoid copying full move_stack when possible (big speedup on deep games)
        try:
            rb = root_board.copy(stack=False)
        except TypeError:
            rb = root_board.copy()
        self.root = MCTSNode(rb)

        if self.verbose:
            print("[MCTS] Starting search...")

        # === Initial expansion ===
        policy, _ = self._model_eval(self.root.board)

        # === Add Dirichlet noise at the root for exploration (if any legal moves) ===
        epsilon = 0.25
        alpha = 0.3
        legal_moves = list(policy.keys())
        if len(legal_moves) > 0:
            try:
                noise = np.random.dirichlet([alpha] * len(legal_moves))
                for i, move in enumerate(legal_moves):
                    policy[move] = (1 - epsilon) * policy[move] + epsilon * noise[i]
                # renormalize
                total = sum(policy.values())
                if total > 0.0:
                    for mv in policy:
                        policy[mv] /= total
            except Exception:
                # if dirichlet fails for any reason, continue without noise
                pass

        # === Expand root node children ===
        for move, p in policy.items():
            try:
                try:
                    next_board = self.root.board.copy(stack=False)
                except TypeError:
                    next_board = self.root.board.copy()
                next_board.push(move)
                self.root.children[move] = MCTSNode(next_board, parent=self.root, prior=p)
            except Exception:
                # if copying or pushing fails, skip that child
                if self.verbose:
                    print(f"[MCTS] Warning: failed to create child for move {move}. Skipping.")

        # === Simulations ===
        for sim in range(self.simulations):
            node = self.root
            search_path = [node]

            # Selection — descend to a leaf
            while node.expanded():
                move, node = self._select_child(node)
                search_path.append(node)

            # Expansion / Evaluation
            terminal, terminal_value = self._is_terminal(node.board)
            if terminal:
                value = terminal_value
            else:
                try:
                    policy, value = self._model_eval(node.board)
                except Exception as e:
                    # model.eval failed unexpectedly; treat node as draw to avoid hang
                    if self.verbose:
                        print(f"[MCTS] Model eval failed: {e}. Treating as draw for this simulation.")
                    value = 0.0
                    policy = {}

                # create children from returned policy (only for legal moves)
                try:
                    for mv, p in policy.items():
                        if mv not in node.children:
                            try:
                                try:
                                    child_board = node.board.copy(stack=False)
                                except TypeError:
                                    child_board = node.board.copy()
                                child_board.push(mv)
                                node.children[mv] = MCTSNode(child_board, parent=node, prior=p)
                            except Exception:
                                # skip moves that fail to copy/push
                                if self.verbose:
                                    print(f"[MCTS] Warning: failed to expand child move {mv}")
                                continue
                except Exception:
                    # defensive: if policy iterable fails, continue
                    pass

            # Backpropagation (negate value up the path)
            for nd in reversed(search_path):
                nd.visit_count += 1
                nd.value_sum += value
                value = -value

            # occasional verbose progress
            if self.verbose and (sim + 1) % 50 == 0:
                print(f"[MCTS] Simulation {sim + 1}/{self.simulations} complete")

        # Build visit distribution at root
        visits = {move: child.visit_count for move, child in self.root.children.items()}
        pi = self._apply_temperature(visits)

        if self.verbose:
            print("[MCTS] Final move probabilities:")
            for mv, prob in sorted(pi.items(), key=lambda x: -x[1]):
                print(f"  {mv}: {round(prob, 3)}")

        return pi

    def _select_child(self, node):
        best_score = -float('inf')
        best_move = None
        best_child = None

        # protect against node.visit_count == 0 (root before any visits)
        parent_visits = max(1, node.visit_count)

        for move, child in node.children.items():
            # UCB-style score with prior
            u = self.c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
            score = child.value() + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def _safe_scalar(self, prob):
        if isinstance(prob, np.ndarray):
            prob = prob.squeeze()
            if getattr(prob, "size", None) != 1:
                raise ValueError(f"Expected scalar, got array with shape {prob.shape}")
            return float(prob.item())
        return float(prob)

    def _model_eval(self, board):
        """
        Evaluate board using model: returns (policy_dict over legal moves, scalar value).
        The policy is a mapping move -> probability (not necessarily normalized first).
        """
        np_tensor = fen_to_tensor(board.fen())
        # ensure shape (1, C, H, W)
        x = torch.tensor(np_tensor, dtype=torch.float32, device=self.device).unsqueeze(0).permute(0, 3, 1, 2)

        with torch.no_grad():
            logits, value = self.model(x)

        probs = F.softmax(logits, dim=1)[0].cpu()
        legal = list(board.legal_moves)
        policy = {}

        for move in legal:
            # try mapping move -> index (some move encoders accept board as second arg)
            try:
                idx = move_to_index(move, board)
            except TypeError:
                idx = move_to_index(move)

            if idx is None:
                # unmapped move — skip
                if self.verbose:
                    pass  # intentionally silent or you can log
                continue

            if not isinstance(idx, int) or idx < 0 or idx >= len(probs):
                continue

            try:
                prob = self._safe_scalar(probs[idx])
            except Exception:
                prob = 0.0
            policy[move] = prob

        # normalize policy; if empty (shouldn't), fallback to uniform over legal moves
        total = sum(policy.values())
        if total > 0.0:
            for mv in policy:
                policy[mv] /= total
        else:
            if len(legal) > 0:
                uniform = 1.0 / len(legal)
                policy = {mv: uniform for mv in legal}
            else:
                policy = {}

        return policy, float(value.item())

    def _apply_temperature(self, visits):
        temp = self.temperature
        if not visits:
            return {}

        if temp == 0:
            best_move = max(visits.items(), key=lambda x: x[1])[0]
            return {move: 1.0 if move == best_move else 0.0 for move in visits}

        # apply temperature to visit counts (common AlphaZero behaviour)
        visits_temp = {move: (count ** (1.0 / temp)) for move, count in visits.items()}
        total = sum(visits_temp.values())
        if total <= 0.0:
            # fallback uniform
            n = len(visits_temp)
            return {move: 1.0 / n for move in visits_temp}
        return {move: prob / total for move, prob in visits_temp.items()}

    def select_move(self, board):
        pi = self.run(board)
        if not pi:
            # fallback: choose any legal move
            legal = list(board.legal_moves)
            return legal[0] if legal else None
        best_move = max(pi.items(), key=lambda x: x[1])[0]
        return best_move

    def get_visit_count_distribution(self, vector_size=4672):
        """
        Returns normalized visit-count policy vector compatible with training (move vector length).
        """
        if not hasattr(self, 'root') or self.root is None or len(self.root.children) == 0:
            return np.zeros(vector_size, dtype=np.float32)

        policy = np.zeros(vector_size, dtype=np.float32)
        total_visits = sum(child.visit_count for child in self.root.children.values())
        if total_visits == 0:
            return policy

        for move, child in self.root.children.items():
            try:
                idx = move_to_index(move, self.root.board)
            except TypeError:
                idx = move_to_index(move)
            if isinstance(idx, int) and 0 <= idx < vector_size:
                policy[idx] = child.visit_count / total_visits

        return policy