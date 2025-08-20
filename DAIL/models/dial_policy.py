# models/dial_policy.py
import math
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------- Utils ---------

class DiagGaussian:
    """
    Simple diagonal Gaussian policy head.
    NOTE: We compute log-probs on the *unsquashed* action.
          In training we clamp actions to the env bounds; this is a common starter simplification.
    """
    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        self.mean = mean
        self.log_std = log_std.clamp(-5, 2)  # keep std in a sane range
        self.std = self.log_std.exp()

    def sample(self) -> torch.Tensor:
        eps = torch.randn_like(self.mean)
        return self.mean + self.std * eps

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        # log N(x | mean, std^2) summed across action dims
        var = self.std.pow(2)
        log_scale = self.log_std
        return (-0.5 * ((x - self.mean) ** 2) / var - log_scale - 0.5 * math.log(2 * math.pi)).sum(-1)

    def entropy(self) -> torch.Tensor:
        # 0.5 * log(2*pi*e*var) per dim, then sum
        return (0.5 + 0.5 * math.log(2 * math.pi) + self.log_std).sum(-1)


def gumbel_softmax_st(logits: torch.Tensor, tau: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Straight-through Gumbel-Softmax: returns (one_hot, soft)
    one_hot is used in the forward pass; soft is used to backprop.
    """
    g = -torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-9)).clamp_min(1e-9))
    y = F.softmax((logits + g) / tau, dim=-1)  # soft sample
    # Straight-through: harden but keep soft grads
    _, ind = y.max(dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y).scatter_(-1, ind, 1.0)
    y_hard = one_hot + (y - y).detach()  # forward=one_hot, backward=soft
    return y_hard, y


# --------- Networks ---------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class DIALPolicy(nn.Module):
    """
    PPO actor-critic with a DIAL message head.

    Inputs:
      obs_dim: env observation dim + incoming message dim (handled by env wrapper)
      act_dim: continuous action dimension (simple_spread_v3 is typically 2)
      K_vocab: number of discrete message tokens

    Forward/act() returns:
      - action (clamped to env bounds)
      - action_logp (log-prob under Gaussian)
      - value (state value)
      - msg_onehot (discrete message to send this step)
      - msg_logits (pre-softmax logits, for logging/entropy)
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        K_vocab: int = 4,
        action_low: float = -1.0,
        action_high: float = 1.0,
        msg_temp_init: float = 1.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.K = K_vocab
        self.a_low = action_low
        self.a_high = action_high
        self.msg_temp = msg_temp_init  # you can anneal this in training

        # Shared torso
        self.backbone = MLP(obs_dim, hidden=128, out_dim=128)

        # Actor head (Gaussian)
        self.mu = nn.Linear(128, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

        # Critic head
        self.v = nn.Linear(128, 1)

        # Message head (logits over K symbols)
        self.msg_head = nn.Linear(128, K_vocab)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Stochastic act for data collection.
        obs: [B, obs_dim]

        Returns dict with:
          action [B, act_dim], action_logp [B], value [B,1],
          msg_onehot [B, K], msg_logits [B, K]
        """
        feats = self.backbone(obs)
        # Message
        msg_logits = self.msg_head(feats)
        msg_onehot, msg_soft = gumbel_softmax_st(msg_logits, tau=self.msg_temp)

        # Action
        mean = self.mu(feats)
        dist = DiagGaussian(mean, self.log_std)
        a_unclipped = dist.sample()
        logp = dist.log_prob(a_unclipped)

        # Clamp to env bounds (starter simplification)
        action = torch.clamp(a_unclipped, min=self.a_low, max=self.a_high)

        value = self.v(feats)
        return {
            "action": action,
            "action_logp": logp,
            "value": value,
            "msg_onehot": msg_onehot,
            "msg_logits": msg_logits,
        }

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        detach_value: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Used for PPO updates with stored actions.
        Returns dict with new logp, entropy, value, message logits (for message entropy).
        """
        feats = self.backbone(obs)

        # Messages (no sampling here; we use logits for an entropy bonus)
        msg_logits = self.msg_head(feats)
        msg_probs = F.softmax(msg_logits, dim=-1)
        msg_entropy = -(msg_probs * (msg_probs.clamp_min(1e-8).log())).sum(-1)  # per-sample

        # Action dist
        mean = self.mu(feats)
        dist = DiagGaussian(mean, self.log_std)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()

        value = self.v(feats)
        if detach_value:
            value = value.detach()

        return {
            "logp": logp,
            "entropy": entropy,
            "value": value.squeeze(-1),
            "msg_logits": msg_logits,
            "msg_entropy": msg_entropy,
        }

    @torch.no_grad()
    def value_only(self, obs: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(obs)
        return self.v(feats).squeeze(-1)
