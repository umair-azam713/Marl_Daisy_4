# HYBRID/models/comm_policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical


def orthogonal_init(layer, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class CommPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, K_vocab=4, hidden_sizes=(128, 128)):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.K_vocab = K_vocab

        layers = []
        last = obs_dim
        for hs in hidden_sizes:
            layers += [nn.Linear(last, hs), nn.Tanh()]
            last = hs
        self.act_body = nn.Sequential(*layers)
        self.mean_head = nn.Linear(last, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

        layers_m = []
        last = obs_dim
        for hs in hidden_sizes:
            layers_m += [nn.Linear(last, hs), nn.Tanh()]
            last = hs
        self.msg_body = nn.Sequential(*layers_m)
        self.msg_head = nn.Linear(last, K_vocab)

        self.apply(lambda m: orthogonal_init(m, gain=1.0))
        orthogonal_init(self.mean_head, gain=0.01)
        orthogonal_init(self.msg_head, gain=0.01)

    def _dist_a(self, obs):
        x = self.act_body(obs)
        mean = self.mean_head(x)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def _dist_m(self, obs):
        x = self.msg_body(obs)
        logits = self.msg_head(x)
        return Categorical(logits=logits), logits

    @torch.no_grad()
    def act(self, obs):
        """
        obs: tensor [batch, obs_dim] or [obs_dim]
        Returns: action, logp_a, msg_idx (int), logp_m, msg_logits
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        dist_a = self._dist_a(obs)
        a = dist_a.sample()
        logp_a = dist_a.log_prob(a).sum(-1)

        dist_m, logits = self._dist_m(obs)
        m = dist_m.sample()
        logp_m = dist_m.log_prob(m)

        return a.squeeze(0), logp_a.squeeze(0), m.squeeze(0).long(), logp_m.squeeze(0), logits.squeeze(0)

    def evaluate_actions(self, obs, act, msg_idx):
        """
        obs: [B, obs_dim]
        act: [B, act_dim]
        msg_idx: [B] long
        Returns dict with exact keys: logp_a, entropy_a, logp_m, entropy_m
        """
        dist_a = self._dist_a(obs)
        logp_a = dist_a.log_prob(act).sum(-1)
        entropy_a = dist_a.entropy().sum(-1)

        dist_m, _ = self._dist_m(obs)
        logp_m = dist_m.log_prob(msg_idx)
        entropy_m = dist_m.entropy()

        return {
            'logp_a': logp_a,
            'entropy_a': entropy_a,
            'logp_m': logp_m,
            'entropy_m': entropy_m,
        }
