import math
from typing import Dict, Optional, Tuple, List

import chess
import numpy as np
import torch
import torch.nn.functional as F

from data.generate_labeled_data import fen_to_tensor
from training.move_encoding import move_to_index, get_total_move_count


# ============================================================
# MCTS Node
# ============================================================

class MCTSNode:
    """
    Един възел от MCTS дървото.

    Всички стойности (value_sum, q_value и т.н.) се интерпретират
    от гледна точка на ИГРАЧА НА ХОД в този възел.
    """

    __slots__ = (
        "board",
        "parent",
        "prior",
        "visit_count",
        "value_sum",
        "children",
        "is_expanded",
        "is_terminal",
        "terminal_value",
    )

    def __init__(self, board: chess.Board, parent: Optional["MCTSNode"] = None, prior: float = 0.0):
        self.board: chess.Board = board
        self.parent: Optional[MCTSNode] = parent
        self.prior: float = float(prior)

        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: Dict[chess.Move, "MCTSNode"] = {}

        self.is_expanded: bool = False
        self.is_terminal: bool = False
        self.terminal_value: float = 0.0  # от гледната точка на играча на ход в този възел

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    # --------------------------------------------------------

    def _compute_terminal_value(self) -> float:
        """
        Връща стойност от гледна точка на ИГРАЧА НА ХОД в тази терминална позиция.
        """
        if not self.board.is_game_over():
            return 0.0

        result = self.board.result()  # '1-0', '0-1', '1/2-1/2', '*'
        side = self.board.turn  # кой е на ход (обикновено губещият при мат)

        if result == "1-0":
            # ако белите са победили
            if side == chess.WHITE:
                return -1.0  # на ход са белите, но вече са загубили (няма ходове)
            else:
                return 1.0
        elif result == "0-1":
            # ако черните са победили
            if side == chess.BLACK:
                return -1.0
            else:
                return 1.0
        else:
            # реми или нещо странно
            return 0.0

    # --------------------------------------------------------

    def expand(self, policy_logits: torch.Tensor):
        """
        Разширява възела, използвайки policy логитите от мрежата.

        policy_logits: 1D тензор с размер get_total_move_count().
        """
        if self.is_expanded:
            return

        # Ако позицията е терминална – няма деца
        if self.board.is_game_over():
            self.is_terminal = True
            self.terminal_value = self._compute_terminal_value()
            self.is_expanded = True
            return

        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            # пат или някаква терминална аномалия
            self.is_terminal = True
            # пат → реми
            self.terminal_value = 0.0
            self.is_expanded = True
            return

        # Нормален случай – използваме policy логитите
        total_moves = get_total_move_count()
        policy = policy_logits.detach().cpu().float().view(-1)

        # Уверяваме се, че policy има достатъчна дължина
        if policy.numel() < total_moves:
            policy = F.pad(policy, (0, total_moves - policy.numel()))
        elif policy.numel() > total_moves:
            policy = policy[:total_moves]

        move_priors = []
        moves_for_priors = []

        for mv in legal_moves:
            idx = move_to_index(mv)
            if idx is None or idx < 0 or idx >= total_moves:
                # safety – ако move_encoding не покрива хода по някаква причина
                continue
            moves_for_priors.append(mv)
            move_priors.append(policy[idx].item())

        # Ако поради някаква причина няма нито един валиден индекс -> униформно
        if not moves_for_priors:
            prob = 1.0 / len(legal_moves)
            for mv in legal_moves:
                child_board = self.board.copy(stack=False)
                child_board.push(mv)
                self.children[mv] = MCTSNode(child_board, parent=self, prior=prob)
            self.is_expanded = True
            return

        logits = torch.tensor(move_priors, dtype=torch.float32)
        probs = torch.softmax(logits, dim=0).numpy()

        for mv, p in zip(moves_for_priors, probs):
            child_board = self.board.copy(stack=False)
            child_board.push(mv)
            self.children[mv] = MCTSNode(child_board, parent=self, prior=float(p))

        self.is_expanded = True


# ============================================================
# MCTS Core
# ============================================================

class MCTS:
    """
    AlphaZero-подобен MCTS с:
    - batched inference
    - Dirichlet noise в root-а
    - динамичен temperature за root policy
    - repetition penalty, check/mate бонуси, stuck penalty
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        simulations: int = 200,
        c_puct: float = 1.8,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.35,
        draw_value_bias: float = -0.15,
        repetition_penalty: float = 0.3,
        aggressive_check_bonus: float = 0.1,
        mate_bonus: float = 0.9,
        stuck_penalty_value: float = 0.05,
        batch_size: int = 64,
    ):
        """
        :param model: AlphaZero модел с forward(x) -> (policy_logits, value)
        :param device: torch.device ("cuda" или "cpu")
        :param simulations: общ брой MCTS симулации
        :param c_puct: PUCT константа
        :param dirichlet_alpha: α за Dirichlet noise
        :param dirichlet_epsilon: коефициент за смесване с noise-а
        :param draw_value_bias: bias към 0 при терминални ремита (леко губещи)
        :param repetition_penalty: наказание за ходове към repetition
        :param aggressive_check_bonus: бонус за ходове, които дават шах
        :param mate_bonus: бонус за ходове водещи до мат
        :param stuck_penalty_value: наказание за позиции с много малко легални ходове
        :param batch_size: размер на batch за inference (leaf възли)
        """
        self.model = model
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct

        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        self.draw_value_bias = draw_value_bias
        self.repetition_penalty = repetition_penalty
        self.aggressive_check_bonus = aggressive_check_bonus
        self.mate_bonus_value = mate_bonus
        self.stuck_penalty_value = stuck_penalty_value

        self.batch_size = max(1, batch_size)
        self.total_moves = get_total_move_count()

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def run(self, root_board: chess.Board, move_number: Optional[int] = None) -> Dict[chess.Move, float]:
        """
        Стартира MCTS от root_board и връща dict {move: prob}.
        """
        root = MCTSNode(root_board.copy(stack=False))
        root_noise_applied = False

        pending: List[Tuple[MCTSNode, List[MCTSNode]]] = []

        self.model.eval()

        for sim in range(self.simulations):
            node = root
            path = [node]

            # -------- Selection --------
            while True:
                if node.is_terminal:
                    break
                if not node.is_expanded or not node.children:
                    break
                move, child = self._select_child(node)
                node = child
                path.append(node)

            pending.append((node, path))

            # Ако имаме достатъчно leaf-ове, или сме на последна симулация
            if len(pending) >= self.batch_size or sim == self.simulations - 1:
                batch_nodes = [n for (n, _) in pending]
                values = self._evaluate_and_expand_batch(batch_nodes)

                # backup с draw bias за терминални ремита
                for (leaf, path_nodes), v in zip(pending, values):
                    v_backup = float(v)
                    if leaf.is_terminal and abs(leaf.terminal_value) < 1e-6:
                        v_backup += self.draw_value_bias
                    self._backup(path_nodes, v_backup)

                pending.clear()

                # Dirichlet noise – веднъж, след като root е expand-нат
                if (not root_noise_applied) and root.is_expanded and root.children:
                    self._add_dirichlet_noise(root, move_number=move_number)
                    root_noise_applied = True

        # Превръщаме посещенията в policy
        return self._build_root_policy(root, move_number=move_number)

    # --------------------------------------------------------
    # Batch evaluation / expansion
    # --------------------------------------------------------

    def _encode_board_batch(self, boards: List[chess.Board]) -> torch.Tensor:
        """
        Кодира списък от board-ове в тензор [B, 15, 8, 8].
        """
        tensors = []
        for b in boards:
            fen = b.fen()
            arr = fen_to_tensor(fen)  # (8,8,15) или (15,8,8)
            arr = np.array(arr, copy=False)

            if arr.shape == (8, 8, 15):
                arr = np.transpose(arr, (2, 0, 1))  # CHW
            elif arr.shape != (15, 8, 8):
                arr = np.reshape(arr, (15, 8, 8))

            tensors.append(torch.tensor(arr, dtype=torch.float32))

        x = torch.stack(tensors, dim=0).to(self.device)  # [B,15,8,8]
        return x

    def _evaluate_and_expand_batch(self, nodes: List[MCTSNode]) -> List[float]:
        """
        Оценява и/или разширява batch от leaf възли.
        Връща списък от value-та за backup (едно за всеки node),
        от гледна точка на ИГРАЧА НА ХОД в съответния възел.
        """
        if not nodes:
            return []

        terminal_indices: List[int] = []
        terminal_values: List[float] = []
        nonterm_indices: List[int] = []
        nonterm_nodes: List[MCTSNode] = []

        # Разделяме на терминални и нетерминални
        for i, node in enumerate(nodes):
            if node.is_terminal or node.board.is_game_over():
                node.is_terminal = True
                node.terminal_value = node._compute_terminal_value()
                node.is_expanded = True

                terminal_indices.append(i)
                terminal_values.append(node.terminal_value)
            else:
                nonterm_indices.append(i)
                nonterm_nodes.append(node)

        values = [0.0] * len(nodes)

        # Нетерминалните – през модела
        if nonterm_nodes:
            boards = [n.board for n in nonterm_nodes]
            x = self._encode_board_batch(boards)
            with torch.no_grad():
                policy_logits_batch, value_batch = self.model(x)

            policy_logits_batch = policy_logits_batch.view(len(nonterm_nodes), -1)
            value_batch = value_batch.view(-1)

            for idx, node, plogits, val in zip(nonterm_indices, nonterm_nodes, policy_logits_batch, value_batch):
                node.expand(plogits)
                v = float(val.item())
                values[idx] = v

        # Терминалните – директно от terminal_value
        for idx, v in zip(terminal_indices, terminal_values):
            values[idx] = float(v)

        return values

    # --------------------------------------------------------
    # Dirichlet noise
    # --------------------------------------------------------

    def _add_dirichlet_noise(self, root: MCTSNode, move_number: Optional[int]):
        """
        Смесва Dirichlet noise в priors на root-а за по-добра експлорация.
        """
        moves = list(root.children.keys())
        if len(moves) <= 1:
            return

        alpha = self.dirichlet_alpha
        eps = self.dirichlet_epsilon

        # По-малко noise в по-късните фази на партията
        if move_number is not None:
            if move_number > 40:
                eps *= 0.25
            elif move_number > 20:
                eps *= 0.5

        noise = np.random.dirichlet([alpha] * len(moves))
        for mv, n in zip(moves, noise):
            child = root.children[mv]
            child.prior = (1.0 - eps) * child.prior + eps * float(n)

    # --------------------------------------------------------
    # Child selection (PUCT + бонуси/наказания)
    # --------------------------------------------------------

    def _select_child(self, node: MCTSNode) -> Tuple[chess.Move, MCTSNode]:
        """
        Избор на дете по PUCT:

            score = Q + U + check_bonus + mate_bonus - rep_pen - stuck_penalty
        """
        best_score = -float("inf")
        best_move = None
        best_child = None

        parent_visits = max(1, node.visit_count)
        sqrt_parent = math.sqrt(parent_visits)

        for mv, child in node.children.items():
            q = child.q_value
            u = self.c_puct * child.prior * (sqrt_parent / (1 + child.visit_count))

            # Наказание за repetition (ограничено – заради липса на пълна история)
            rep_pen = 0.0
            try:
                if child.board.can_claim_threefold_repetition():
                    rep_pen = self.repetition_penalty
            except Exception:
                rep_pen = 0.0

            # Бонус за шах
            check_bonus = 0.0
            try:
                if child.board.is_check():
                    check_bonus = self.aggressive_check_bonus
            except Exception:
                check_bonus = 0.0

            # Бонус за мат
            mate_bonus = 0.0
            try:
                if child.board.is_game_over() and child.board.is_checkmate():
                    mate_bonus = self.mate_bonus_value
            except Exception:
                mate_bonus = 0.0

            # Stuck penalty – много малко легални ходове
            stuck_penalty = 0.0
            try:
                num_legal = child.board.legal_moves.count()
                if num_legal <= 2:
                    stuck_penalty = self.stuck_penalty_value
            except Exception:
                stuck_penalty = 0.0

            score = q + u + check_bonus + mate_bonus - rep_pen - stuck_penalty

            if score > best_score:
                best_score = score
                best_move = mv
                best_child = child

        if best_child is None:
            # safety fallback
            best_move, best_child = next(iter(node.children.items()))

        return best_move, best_child

    # --------------------------------------------------------
    # Backup
    # --------------------------------------------------------

    def _backup(self, path: List[MCTSNode], leaf_value: float):
        """
        Backup на leaf_value по пътя от листа към root.
        leaf_value е от гледна точка на страната на ход в LEAF възела.
        На всяка стъпка сменяме знака (сменя се гледната точка).
        """
        v = float(leaf_value)
        for node in reversed(path):
            node.value_sum += v
            node.visit_count += 1
            v = -v  # сменяме гледната точка (играчите се редуват)

    # --------------------------------------------------------
    # Root policy
    # --------------------------------------------------------

    def _build_root_policy(self, root: MCTSNode, move_number: Optional[int]) -> Dict[chess.Move, float]:
        """
        Превръща visit count-овете на root-а в dict {move: prob}
        чрез softmax с динамична температура.
        """
        if not root.children:
            return {}

        moves = list(root.children.keys())
        visits = np.array([root.children[mv].visit_count for mv in moves], dtype=np.float32)

        if visits.sum() <= 0:
            # fallback към priors
            priors = np.array([root.children[mv].prior for mv in moves], dtype=np.float32)
            if priors.sum() <= 0:
                priors = np.ones_like(priors)
            priors /= priors.sum()
            return {mv: float(p) for mv, p in zip(moves, priors)}

        # Динамичен temperature schedule
        if move_number is None:
            temperature = 1.0
        elif move_number < 10:
            temperature = 1.0
        elif move_number < 30:
            temperature = 0.7
        elif move_number < 50:
            temperature = 0.4
        else:
            temperature = 0.2

        if temperature <= 1e-6:
            best_idx = int(np.argmax(visits))
            pi = np.zeros_like(visits)
            pi[best_idx] = 1.0
        else:
            scaled = np.power(visits, 1.0 / max(temperature, 1e-6))
            if scaled.sum() <= 0:
                scaled = np.ones_like(scaled)
            pi = scaled / scaled.sum()

        return {mv: float(p) for mv, p in zip(moves, pi)}