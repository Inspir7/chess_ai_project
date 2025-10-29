# training/self_play_trainer.py
import os
import torch
import numpy as np
from training.buffer import ReplayBuffer
from training import move_encoding  # ползваме твоето move_encoding.py

class SelfPlayTrainer:
    """
    Simple trainer class for self-play / online training.
    Responsibilities:
      - hold model, optimizer, buffer
      - convert pi_dict -> full policy vector using move_encoding
      - train from buffer (single batch or multiple steps)
      - save/load checkpoints
    """
    def __init__(self, model, device='cpu', buffer_size=5000, lr=1e-3):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model = model.to(self.device)
        self.buffer = ReplayBuffer(max_size=buffer_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # expected policy dimension
        self.total_moves = move_encoding.get_total_move_count()

    def pi_dict_to_vector(self, pi_dict):
        """
        Convert {chess.Move: prob, ...} into fixed-size numpy vector (total_moves).
        Unknown moves (not in mapping) are ignored.
        Normalizes vector to sum=1 if sum>0.
        """
        vec = np.zeros(self.total_moves, dtype=np.float32)
        for mv, p in pi_dict.items():
            idx = move_encoding.move_to_index(mv)
            if idx is None or idx < 0 or idx >= self.total_moves:
                continue
            vec[idx] = float(p)
        s = vec.sum()
        if s > 0:
            vec /= s
        return vec

    def push_from_pi(self, state_tensor, pi_dict, value=None):
        """
        state_tensor: numpy or torch (C,H,W) or (H,W,C)
        pi_dict: dict mapping chess.Move -> probability
        value: float or None
        """
        pi_vec = self.pi_dict_to_vector(pi_dict)
        # store state as numpy or torch (we keep as-is; ReplayBuffer handles conversion)
        self.buffer.push(state_tensor, pi_vec, value)

    def train_from_buffer(self, batch_size=32):
        """
        Sample a batch and train one step.
        Returns (policy_loss, value_loss, total_loss) as floats.
        """
        if len(self.buffer) < 1:
            return 0.0, 0.0, 0.0
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        states, policies, values = self.buffer.sample_as_tensors(batch_size, device=self.device)

        # Ensure states shape is (B,C,H,W) and C is model-expected (we assume model handles it)
        if states.ndim == 4 and states.shape[1] not in (1,3,15):
            # try channels-last -> channels-first
            states = states.permute(0,3,1,2)

        self.model.train()
        self.optimizer.zero_grad()
        logits, pred_value = self.model(states)  # logits: (B, total_moves), pred_value: (B,1) or (B,)
        # policy loss: cross-entropy like (we have targets as prob distributions)
        log_probs = torch.log_softmax(logits, dim=1)
        policy_loss = -torch.sum(policies * log_probs) / states.size(0)
        # value loss: MSE
        value_loss = torch.mean((pred_value.squeeze() - values) ** 2)
        total_loss = policy_loss + value_loss
        total_loss.backward()
        self.optimizer.step()

        return float(policy_loss.item()), float(value_loss.item()), float(total_loss.item())

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }, path)

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])