import torch.nn as nn
import torch

class ValueHead(nn.Module):
    def __init__(self, input_channels=64, board_size=8):
        super(ValueHead, self).__init__()
        self.conv = nn.Conv2d(input_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(board_size * board_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.tanh(x)  # стойност между -1 и 1
