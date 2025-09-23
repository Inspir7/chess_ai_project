import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ChessTensorDataset(Dataset):
    def __init__(self, directory):
        self.samples = []

        for file in os.listdir(directory):
            if file.endswith('.npy'):
                path = os.path.join(directory, file)
                data = np.load(path, allow_pickle=True)
                self.samples.extend(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        state = torch.tensor(sample["state"], dtype=torch.float32)
        policy = torch.tensor(sample["policy"], dtype=torch.float32)
        value = torch.tensor(sample["value"], dtype=torch.float32)

        return state, policy, value
