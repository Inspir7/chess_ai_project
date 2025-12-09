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
    def __init__(self, max_size: int = 50000):
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
        states_list, policies_list, values_tensor = self.sample(batch_size)

        # States -> (B, C, H, W)
        state_tensors = []
        for s in states_list:
            st = s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32)
            if st.ndim == 3 and st.shape[0] not in (1, 3, 15):
                st = st.permute(2, 0, 1)
            state_tensors.append(st)

        states = torch.stack(state_tensors).float()

        # Policies -> (B, 4672)
        policies = torch.stack([
            p if isinstance(p, torch.Tensor) else torch.tensor(p, dtype=torch.float32)
            for p in policies_list
        ])

        # Values should be (B,)
        values = values_tensor.view(-1).float()

        if device is not None:
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)

        return states, policies, values

    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)