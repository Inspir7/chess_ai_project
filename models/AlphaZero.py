import torch.nn as nn
from models.encoder import Encoder
from models.policy_head import PolicyHead
from models.value_head import ValueHead

class AlphaZeroModel(nn.Module):
    def __init__(self):
        super(AlphaZeroModel, self).__init__()
        self.encoder = Encoder()
        self.policy_head = PolicyHead()
        self.value_head = ValueHead()

    def forward(self, x):
        x = self.encoder(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
