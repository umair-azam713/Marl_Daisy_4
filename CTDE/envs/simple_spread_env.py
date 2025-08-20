# envs/simple_spread_env.py
import numpy as np
from gymnasium import spaces
from pettingzoo.mpe import simple_spread_v3


class SimpleSpreadCTDE:
    """
    Simple Spread with explicit (optional) communication for CTDE, using the PARALLEL API.
    - Agents observe: base obs + incoming one-hot messages from others (if provided).
    - Central critic can call get_state() to get a global state (all base obs + all msgs).
    """

    def __init__(self, num_agents=3, max_cycles=50, continuous_actions=True, K_vocab=4):
        self.num_agents = num_agents
        self.K_vocab = K_vocab

        # >>> USE PARALLEL API <<<
        self.env = simple_spread_v3.parallel_env(
            N=num_agents,
            local_ratio=0.5,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions
        )
        # Will be populated after reset()
        self.agents = []

        # Init message memory
        self._last_msgs = {}

    # ---------- API ----------

    def reset(self, seed=None):
        # Gymnasium-style: (obs_dict, infos)
        result = self.env.reset(seed=seed)
        if isinstance(result, tuple):
            obs_dict, _infos = result
        else:
            obs_dict = result  # fallback

        self.agents = list(self.env.agents)
        self._last_msgs = {agent: np.zeros(self.K_vocab, dtype=np.float32) for agent in self.agents}
        return self._concat_msgs(obs_dict)

    def step(self, action_dict, msg_dict=None):
        """
        action_dict: dict(agent -> np.ndarray(float32) matching action_space(agent).shape)
        msg_dict:    dict(agent -> int index in [0, K_vocab-1]) OR one-hot np.ndarray; optional
        """
        # Ensure a valid action for every agent, correct dtype/shape, and within bounds
        actions = {}
        for ag in self.env.agents:
            if ag in action_dict:
                a = np.asarray(action_dict[ag], dtype=np.float32)
            else:
                # default to zeros if missing
                sample = self.env.action_space(ag).sample()
                a = np.zeros_like(sample, dtype=np.float32)
            # sanitize
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            # clip to space bounds and enforce shape
            low = self.env.action_space(ag).low
            high = self.env.action_space(ag).high
            a = np.clip(a, low, high).astype(np.float32).reshape(self.env.action_space(ag).shape)
            actions[ag] = a

        # Parallel step: returns (obs, rewards, terminations, truncations, infos)
        obs_dict, rewards, terminations, truncations, infos = self.env.step(actions)

        # Update stored messages for next-step observations
        new_msgs = {}
        for ag in self.env.agents:
            onehot = np.zeros(self.K_vocab, dtype=np.float32)
            if msg_dict is not None and ag in msg_dict:
                v = msg_dict[ag]
                if isinstance(v, (int, np.integer)):
                    if 0 <= v < self.K_vocab:
                        onehot[v] = 1.0
                else:
                    # treat as one-hot vector
                    v = np.asarray(v, dtype=np.float32).reshape(-1)
                    if v.shape[0] == self.K_vocab:
                        # sanitize one-hot-ish
                        idx = int(np.argmax(v))
                        onehot[idx] = 1.0
            new_msgs[ag] = onehot
        self._last_msgs = new_msgs

        dones = {ag: bool(terminations.get(ag, False) or truncations.get(ag, False)) for ag in self.env.agents}
        obs_with_msgs = self._concat_msgs(obs_dict)
        return obs_with_msgs, rewards, dones, infos

    # ---------- Helpers ----------

    def _concat_msgs(self, obs_dict):
        out = {}
        # Build obs = base_obs + incoming msgs from others
        for i, ag in enumerate(self.env.agents):
            incoming = []
            for j, other in enumerate(self.env.agents):
                if other != ag:
                    incoming.append(self._last_msgs.get(other, np.zeros(self.K_vocab, dtype=np.float32)))
            if incoming:
                incoming_vec = np.concatenate(incoming, axis=0).astype(np.float32)
                out[ag] = np.concatenate([obs_dict[ag].astype(np.float32), incoming_vec], axis=0)
            else:
                out[ag] = obs_dict[ag].astype(np.float32)
        return out

    def get_state(self, obs_with_msgs_dict):
        """
        Global state for centralized critic: concat all base observations (without incoming msgs)
        + concat all agents' last sent messages (one-hot).
        """
        # Derive base obs sizes from current obs (strip incoming msgs)
        base_obs_list = []
        for ag in self.env.agents:
            # base obs = total obs - incoming_msgs_dim
            total = obs_with_msgs_dict[ag].shape[0]
            comm_dim = (self.num_agents - 1) * self.K_vocab
            base_obs = obs_with_msgs_dict[ag][: total - comm_dim] if total > comm_dim else obs_with_msgs_dict[ag]
            base_obs_list.append(base_obs.astype(np.float32))

        msg_list = [self._last_msgs.get(ag, np.zeros(self.K_vocab, dtype=np.float32)) for ag in self.env.agents]
        return np.concatenate(base_obs_list + msg_list, axis=0).astype(np.float32)

    # Spaces reflecting agent observations incl. messages
    def observation_space(self, agent):
        base_dim = self.env.observation_space(agent).shape[0]
        comm_dim = (self.num_agents - 1) * self.K_vocab
        return spaces.Box(low=-np.inf, high=np.inf, shape=(base_dim + comm_dim,), dtype=np.float32)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def close(self):
        self.env.close()
