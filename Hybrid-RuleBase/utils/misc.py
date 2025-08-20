# HYBRID/utils/misc.py
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


def safe_mean(xs):
    return float(np.mean(xs)) if len(xs) else 0.0


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards, values, dones are 1D numpy arrays of length T
    returns advantages and returns-to-go arrays (length T)
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - (1.0 if (t + 1 < T and dones[t + 1]) else 0.0)
        nextvalue = values[t + 1] if (t + 1 < T) else 0.0
        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    return adv, returns