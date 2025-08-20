# models/central_critic.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CentralCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.v_out = nn.Linear(256, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.v_out(x)
        return v.squeeze(-1)  # [B]
