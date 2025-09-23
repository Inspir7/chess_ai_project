import numpy as np
import torch
from torch.utils.data import Dataset

class ChessLabeledTensorDataset(Dataset):
    def __init__(self, states_dir, policies_dir, values_dir):
        self.states = np.load(states_dir)
        self.policies = np.load(policies_dir)
        self.values = np.load(values_dir)

        assert len(self.states) == len(self.policies) == len(self.values), "Размерите на масивите не съвпадат."

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = torch.tensor(self.states[idx], dtype=torch.float32).permute(2, 0, 1)  # ⬅️ от (8, 8, 15) в (15, 8, 8)
        policy = torch.tensor(self.policies[idx], dtype=torch.float32)
        value = torch.tensor(self.values[idx], dtype=torch.float32)
        return state, policy, value
