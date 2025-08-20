# utils/misc.py
import numpy as np
import torch
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(x, device="cpu"):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def compute_gae(rews, dones, values, gamma=0.99, lam=0.95):
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - float(dones[t])
        nextvalue = values[t + 1] if t + 1 < T else 0.0
        delta = rews[t] + gamma * nextvalue * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values[:T]
    return adv, returns


def safe_mean(xs):
    return float(np.mean(xs)) if len(xs) else 0.0
