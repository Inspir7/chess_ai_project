# training/RL_pipeline.py
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from datetime import datetime

from models.AlphaZero import AlphaZeroModel
# move_encoding може да има различни подписи; ще работим defensively по-долу
from models.move_encoding import index_to_move, move_to_index
from training.mcts import MCTS
from utils.chess_utils import initial_board, board_to_tensor, game_over, result_from_perspective

# -----------------------
# Replay buffer (simple, stores tuples)
# -----------------------
class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, examples):
        """
        examples: iterable of (state_tensor_or_np, policy_vec_or_tensor, value_scalar)
        We store them as-is; sample() returns stacked tensors.
        """
        self.buffer.extend(examples)

    def sample(self, batch_size):
        """
        Returns (states_tensor (B,C,H,W), policies_tensor (B,move_dim), values_tensor (B,1))
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)

        # Convert states -> tensors, ensure channel-first (C,H,W)
        states_t = []
        for s in states:
            if isinstance(s, torch.Tensor):
                st = s.detach().cpu().float()
            else:
                st = torch.tensor(s, dtype=torch.float32)
            # If HWC -> CHW
            if st.ndim == 3 and st.shape[0] not in (1, 3, 15) and st.shape[-1] in (1,3,15):
                st = st.permute(2, 0, 1)
            # If still channels-last e.g. (8,8,15)
            if st.ndim == 3 and st.shape[0] not in (1,3,15) and st.shape[-1] == 15:
                st = st.permute(2,0,1)
            states_t.append(st)

        # Convert policies -> tensors (assume already full-length vectors)
        policies_t = []
        for p in policies:
            if isinstance(p, torch.Tensor):
                policies_t.append(p.detach().cpu().float())
            else:
                policies_t.append(torch.tensor(p, dtype=torch.float32))

        # Values: replace None with 0.0 (bootstrap)
        values_t = torch.tensor([0.0 if (v is None) else float(v) for v in values], dtype=torch.float32).unsqueeze(1)

        # Stack
        states_stack = torch.stack(states_t)
        policies_stack = torch.stack(policies_t)

        return states_stack, policies_stack, values_t

    def __len__(self):
        return len(self.buffer)

# -----------------------
# RL pipeline
# -----------------------
class RLPipeline:
    def __init__(self,
                 model_path="training/alpha_zero_rl.pth",
                 device=None,
                 lr=1e-3,
                 weight_decay=1e-4,
                 buffer_capacity=200_000,
                 move_vector_size=4672):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AlphaZeroModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.model_path = model_path
        self.move_vector_size = move_vector_size

        if os.path.exists(self.model_path):
            print(f"[INFO] Loading existing model from {self.model_path}")
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        else:
            print("[INFO] Starting with new model.")

    # defensive wrapper for move_to_index (some implementations expect (move) others (move, board))
    def _move_to_index(self, move, board):
        try:
            idx = move_to_index(move)
            return idx
        except TypeError:
            try:
                idx = move_to_index(move, board)
                return idx
            except Exception:
                return None
        except Exception:
            return None

    # defensive wrapper for index_to_move (some variants may need board)
    def _index_to_move(self, index, board):
        try:
            mv = index_to_move(index)
            if mv is None:
                # try variant with board
                try:
                    mv = index_to_move(index, board)
                except Exception:
                    mv = None
            return mv
        except TypeError:
            try:
                mv = index_to_move(index, board)
                return mv
            except Exception:
                return None
        except Exception:
            return None

    def self_play_episode(self, mcts_sims=100, temperature=1.0, verbose=False):
        """
        Runs one self-play game using current model + MCTS.
        Returns examples list of (state_tensor [C,H,W], pi_tensor [move_dim], z scalar)
        and move_logs for debugging.
        """
        board = initial_board()
        player_color = board.turn
        #ako iskam postoqnni logove ot simulaciqta -> verbose=verbose/True
        mcts = MCTS(self.model, device=self.device, simulations=mcts_sims, temperature=temperature, verbose=False)

        game_history = []
        move_logs = []
        step = 0
        MAX_STEPS = 1000

        while not game_over(board) and step < MAX_STEPS:
            pi_dict = mcts.run(board)  # dict: chess.Move -> probability
            # build policy vector (torch)
            pi = torch.zeros(self.move_vector_size, dtype=torch.float32)

            for mv, prob in pi_dict.items():
                idx = self._move_to_index(mv, board)
                if idx is None:
                    # log once (include FEN to help debug)
                    print(f"[DEBUG] move_to_index returned None for move {mv} on FEN: {board.fen()}")
                    continue
                if 0 <= idx < self.move_vector_size:
                    pi[idx] = float(prob)
                else:
                    print(f"[DEBUG] index out of range: {idx} for move {mv}")

            # normalize (numerical safety)
            total = pi.sum().item()
            if total > 0:
                pi = pi / total
            else:
                # fallback: uniform over legal moves
                legal = list(board.legal_moves)
                if len(legal) == 0:
                    break
                uniform = 1.0 / len(legal)
                for mv in legal:
                    idx = self._move_to_index(mv, board)
                    if idx is not None and 0 <= idx < self.move_vector_size:
                        pi[idx] = uniform

            # state tensor: ensure C,H,W
            st = board_to_tensor(board)
            if not isinstance(st, torch.Tensor):
                st = torch.tensor(st, dtype=torch.float32)
            if st.ndim == 3 and st.shape[-1] == 15:  # H,W,C -> C,H,W
                st = st.permute(2, 0, 1)

            # save
            game_history.append((st, pi, 0.0))

            # logging
            best_idx = int(torch.argmax(pi).item())
            best_mv = self._index_to_move(best_idx, board)
            move_logs.append({"step": step, "best_index": best_idx, "best_move": best_mv, "best_prob": float(pi[best_idx].item())})

            # sample action from pi with temperature
            if temperature == 0:
                chosen_idx = int(torch.argmax(pi).item())
            else:
                probs = pi.numpy()
                # protect against negative/NaN
                probs = np.clip(probs, a_min=0.0, a_max=None)
                sm = probs / (probs.sum() + 1e-12)
                choices = list(range(len(sm)))
                chosen_idx = random.choices(choices, weights=sm, k=1)[0]

            mv = self._index_to_move(chosen_idx, board)
            if mv is None:
                # fallback pick random legal move
                legal = list(board.legal_moves)
                if len(legal) == 0:
                    break
                mv = random.choice(legal)

            if mv in board.legal_moves:
                board.push(mv)
            else:
                # safety fallback: push a legal random move
                legal = list(board.legal_moves)
                if len(legal) == 0:
                    break
                board.push(random.choice(legal))

            step += 1

        z = result_from_perspective(board, player_color)
        examples = [(s, p, z) for (s, p, _) in game_history]
        return examples, move_logs

    def add_examples(self, examples):
        self.buffer.push(examples)

    def train_step(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return None

        states, policies, values = self.buffer.sample(batch_size)
        states = states.to(self.device)
        policies = policies.to(self.device)
        values = values.to(self.device)

        self.model.train()
        out_policy, out_value = self.model(states)
        target_classes = torch.argmax(policies, dim=1)

        loss_p = self.policy_loss_fn(out_policy, target_classes)
        loss_v = self.value_loss_fn(out_value, values)
        loss = loss_p + loss_v

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
        self.optimizer.step()

        return {"loss": loss.item(), "loss_p": loss_p.item(), "loss_v": loss_v.item()}

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"[INFO] Model saved to {self.model_path}")

    def run_training(self, episodes=50, train_steps=20, batch_size=64, sims=100, verbose=False):
        """
        Главен RL loop: за епизод -> self-play -> добавяне в буфер -> няколко train steps -> запис
        """
        import numpy as np  # локално (само ако не е импортнато горе)
        print(f"[RL] Starting training for {episodes} episodes...")
        for ep in range(1, episodes + 1):
            print(f"\n[EP {ep}] Self-play in progress...")
            examples, logs = self.self_play_episode(mcts_sims=sims, temperature=1.0, verbose=verbose)
            self.add_examples(examples)
            print(f"[EP {ep}] Collected {len(examples)} examples (Buffer size: {len(self.buffer)})")

            # optional: small debug print of last few moves
            if verbose:
                for x in logs[-3:]:
                    print(f"  [log] step={x['step']} best_move={x['best_move']} prob={x['best_prob']:.4f}")

            for step in range(train_steps):
                stats = self.train_step(batch_size)
                if stats:
                    print(f"[TRAIN] Step {step+1}/{train_steps} | Loss={stats['loss']:.4f} | Policy={stats['loss_p']:.4f} | Value={stats['loss_v']:.4f}")
                else:
                    print("[TRAIN] Not enough data in buffer yet.")

            # save periodically
            self.save_model()

        print("[RL] Training complete!")

# -----------------------
# quick run when executed directly
# -----------------------
if __name__ == "__main__":
    import numpy as np
    pipeline = RLPipeline(
        model_path="/home/presi/projects/chess_ai_project/training/alpha_zero_supervised.pth",
        move_vector_size=4672  # adjust if your move encoding differs
    )
    pipeline.run_training(episodes=10, train_steps=15, batch_size=64, sims=50, verbose=True)
