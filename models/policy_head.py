import torch.nn as nn

class PolicyHead(nn.Module):
    def __init__(self, input_channels=64, board_size=8):
        super(PolicyHead, self).__init__()
        self.conv = nn.Conv2d(input_channels, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(2 * board_size * board_size, 4672)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)
