# HYBRID/models/central_critic.py
import torch
import torch.nn as nn


def orthogonal_init(layer, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class CentralCritic(nn.Module):
    def __init__(self, state_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        last = state_dim
        for hs in hidden_sizes:
            layers += [nn.Linear(last, hs), nn.Tanh()]
            last = hs
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)
        self.apply(lambda m: orthogonal_init(m, gain=1.0))
        orthogonal_init(self.net[-1], gain=1.0)

    def forward(self, state):
        return self.net(state).squeeze(-1)