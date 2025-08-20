# models/comm_policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CommPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, K_vocab, action_low=-1.0, action_high=1.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.K_vocab = K_vocab
        self.a_low = action_low
        self.a_high = action_high

        # Shared encoder for observations
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        # Action head (continuous)
        self.mu = nn.Linear(128, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Message head (categorical logits)
        self.msg_logits = nn.Linear(128, K_vocab)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        mu = torch.tanh(self.mu(x))  # [-1, 1]
        std = torch.exp(self.log_std)
        msg_logits = self.msg_logits(x)

        return mu, std, msg_logits

    def act(self, obs):
        """
        obs: [B, obs_dim]
        Returns: action, logp_action, msg_idx, msg_onehot, logp_msg
        """
        mu, std, msg_logits = self.forward(obs)

        # Action sampling (Gaussian)
        dist_a = torch.distributions.Normal(mu, std)
        action = dist_a.sample()
        logp_a = dist_a.log_prob(action).sum(axis=-1)
        action_clamped = torch.clamp(action, self.a_low, self.a_high)

        # Message sampling (Categorical)
        dist_m = torch.distributions.Categorical(logits=msg_logits)
        msg_idx = dist_m.sample()
        logp_m = dist_m.log_prob(msg_idx)
        msg_onehot = F.one_hot(msg_idx, num_classes=self.K_vocab).float()

        return {
            "action": action_clamped,
            "action_logp": logp_a,
            "msg_idx": msg_idx,
            "msg_onehot": msg_onehot,
            "msg_logp": logp_m,
            "msg_logits": msg_logits
        }

    def evaluate_actions(self, obs, act, msg_idx):
        """
        For PPO update — evaluates log-probs & entropy for both actions and messages.
        """
        mu, std, msg_logits = self.forward(obs)

        # Action dist
        dist_a = torch.distributions.Normal(mu, std)
        logp_a = dist_a.log_prob(act).sum(axis=-1)
        entropy_a = dist_a.entropy().sum(axis=-1)

        # Message dist
        dist_m = torch.distributions.Categorical(logits=msg_logits)
        logp_m = dist_m.log_prob(msg_idx)
        entropy_m = dist_m.entropy()

        return {
            "logp_a": logp_a,
            "entropy_a": entropy_a,
            "logp_m": logp_m,
            "entropy_m": entropy_m
        }
