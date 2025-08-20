# utils/replay_buffer.py
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional

class ReplayBuffer:
    """
    A simple FIFO replay buffer for off-policy algorithms (e.g., DQN/MADDPG).
    Not used by the current PPO trainer, but handy if you extend the project.

    Stores transitions of the form:
      (obs, action, reward, next_obs, done)
    Optionally supports per-agent dictionaries by providing 'agent' key on add().
    """
    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 100_000, dtype=np.float32):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=dtype)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=dtype)
        self.actions = np.zeros((capacity, act_dim), dtype=dtype)
        self.rews = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rews[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        assert self.size > 0, "ReplayBuffer is empty"
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rews[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.size


class RolloutBufferPPO:
    """
    A lightweight rollout storage for PPO.
    Stores a *single* trajectory (or concatenated mini-trajectories) for updates.

    Use:
      buf = RolloutBufferPPO(obs_dim, act_dim, max_steps)
      for t in steps:
         buf.store(obs, act, logp, value, reward, done)
      adv, ret = buf.compute_gae(gamma, lam)
      batches = buf.iter_minibatches(minibatches)
    """
    def __init__(self, obs_dim: int, act_dim: int, max_steps: int):
        self.max_steps = max_steps
        self.obs = np.zeros((max_steps, obs_dim), dtype=np.float32)
        self.act = np.zeros((max_steps, act_dim), dtype=np.float32)
        self.logp = np.zeros((max_steps,), dtype=np.float32)
        self.val = np.zeros((max_steps + 1,), dtype=np.float32)  # last value for bootstrap
        self.rew = np.zeros((max_steps,), dtype=np.float32)
        self.done = np.zeros((max_steps,), dtype=np.bool_)
        self.t = 0

    def store(self, obs, act, logp, value, reward, done):
        assert self.t < self.max_steps, "RolloutBufferPPO full — increase max_steps"
        self.obs[self.t] = obs
        self.act[self.t] = act
        self.logp[self.t] = logp
        self.val[self.t] = value
        self.rew[self.t] = reward
        self.done[self.t] = done
        self.t += 1

    def finish_path(self, last_value: float = 0.0):
        # store bootstrap value at t=T
        self.val[self.t] = last_value

    def compute_gae(self, gamma: float, lam: float):
        T = self.t
        adv = np.zeros(T, dtype=np.float32)
        lastgaelam = 0.0
        for i in reversed(range(T)):
            next_nonterminal = 1.0 - float(self.done[i])
            delta = self.rew[i] + gamma * self.val[i + 1] * next_nonterminal - self.val[i]
            lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
            adv[i] = lastgaelam
        ret = adv + self.val[:T]
        # normalize advantages here or in trainer
        return adv, ret

    def iter_minibatches(self, minibatches: int):
        T = self.t
        idx = np.arange(T)
        mb_size = max(1, T // minibatches)
        np.random.shuffle(idx)
        for start in range(0, T, mb_size):
            mb_idx = idx[start:start + mb_size]
            yield (
                self.obs[mb_idx],
                self.act[mb_idx],
                self.logp[mb_idx],
                mb_idx,  # indices for aligning adv/ret computed outside
            )

    def clear(self):
        self.t = 0
