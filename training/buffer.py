# training/replay_buffer.py
import random
import numpy as np
import torch

class ReplayBuffer:
    """
    Simple Replay Buffer that stores tuples:
      (state_tensor, policy_vector, value_scalar)
    - state_tensor: torch tensor or numpy array (C,H,W) or (H,W,C)
    - policy_vector: numpy array or list of fixed length (total moves)
    - value_scalar: float (can be None until game end, but training uses 0.0 if None)

    API:
      push(state, policy_vector, value)
      sample(batch_size) -> list of (state, policy_vector, value)
      sample_as_tensors(batch_size, device) -> (states, policies, values)
      clear()
      __len__()
    """
    def __init__(self, max_size: int = 5000):
        self.max_size = max_size
        self.buffer = []

    def push(self, state, policy_vector, value):
        """
        Append a sample. State/policy can be numpy or torch.
        We store as they come (but sample_as_tensors will convert).
        """
        if len(self.buffer) >= self.max_size:
            # pop oldest
            self.buffer.pop(0)
        self.buffer.append((state, policy_vector, value))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)

        # Всички state са 3D Tensor (15x8x8)
        states = [s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32) for s in states]
        policies = [p if isinstance(p, torch.Tensor) else torch.tensor(p, dtype=torch.float32) for p in policies]
        values = torch.tensor(values, dtype=torch.float32).unsqueeze(1)

        return states, policies, values

    def sample_as_tensors(self, batch_size: int, device=None):
        """
        Return:
          states: torch.Tensor (B, C, H, W)
          policies: torch.Tensor (B, total_moves)
          values: torch.Tensor (B,)
        This helper expects stored policy_vectors to already be full-size vectors.
        """
        batch = self.sample(batch_size)
        # States: may be stored as numpy arrays (H,W,C) or (C,H,W). Normalize to (B,C,H,W)
        state_tensors = []
        for s, _, _ in batch:
            # Accept numpy or torch
            if isinstance(s, torch.Tensor):
                st = s.detach().cpu()
            else:
                st = torch.tensor(s, dtype=torch.float32)
            # If channels last (H,W,C) try to permute to (C,H,W)
            if st.ndim == 3 and st.shape[0] not in (1,3,15):
                # assume HWC -> CHW
                st = st.permute(2, 0, 1)
            state_tensors.append(st)

        states = torch.stack(state_tensors).float()
        # Policies
        policy_tensors = []
        for _, p, _ in batch:
            if isinstance(p, torch.Tensor):
                policy_tensors.append(p.detach().cpu().float())
            else:
                policy_tensors.append(torch.tensor(p, dtype=torch.float32))
        policies = torch.stack(policy_tensors)

        # Values: replace None with 0.0 (bootstrap) for training; caller can set differently
        values = torch.tensor([0.0 if (v is None) else float(v) for _, _, v in batch], dtype=torch.float32)

        if device is not None:
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)

        return states, policies, values

    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)