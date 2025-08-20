import numpy as np
from pettingzoo.mpe import simple_spread_v3

class SimpleSpreadHybridWrapper:
    """
    Parallel API wrapper over PettingZoo simple_spread_v3 that augments observations with a
    fixed-size incoming message section and handles rule-based enabling/disabling of comms.

    - step(actions_dict, msg_dict_or_none, comm_on: bool)
      * messages can be int indices or one-hot arrays per agent; stored as one-hot
      * if comm_on is False, we still append zero-padded message section to each obs
    - get_state(obs_with_msgs_dict) -> global state (concat base obs of all agents + last msgs of all agents)

    Reward shaping added:
      * Encourage spreading (avg pairwise distance > 1.0 → +0.05)
      * Penalize collisions if infos reports them (–0.1)
      * Encourage landmark coverage (+1.0 if agent close to landmark)
    """

    def __init__(self, N=3, local_ratio=0.5, continuous_actions=True,
                 max_cycles=25, K_vocab=4, seed=42):
        self._env = simple_spread_v3.parallel_env(
            N=N, local_ratio=local_ratio, continuous_actions=continuous_actions, max_cycles=max_cycles
        )
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._env.reset(seed=seed)
        self.agents = list(self._env.agents)
        self.num_agents = len(self.agents)
        self.K_vocab = int(K_vocab)

        # determine base obs dim from underlying env
        sample_obs = self._env.observe(self.agents[0]) if hasattr(self._env, 'observe') else None
        if sample_obs is None:
            obs_dict, _ = self._env.reset(seed=seed)
            sample_obs = obs_dict[self.agents[0]]
        self.base_obs_dim = int(sample_obs.shape[0])
        # per-agent incoming message dim = (num_agents-1) * K
        self.incoming_msg_dim = (self.num_agents - 1) * self.K_vocab
        self.total_obs_dim = self.base_obs_dim + self.incoming_msg_dim

        # storage for last sent messages (one-hot) per agent; used to build incoming message section for others
        self._last_msgs = {agent: np.zeros((self.K_vocab,), dtype=np.float32) for agent in self.agents}

    def reset(self, seed=None, options=None):
        obs_dict, infos = self._env.reset(seed=seed if seed is not None else self._seed, options=options)
        # zero messages at reset
        self._last_msgs = {agent: np.zeros((self.K_vocab,), dtype=np.float32) for agent in self.agents}
        # build obs with zeroed message part
        obs_aug = {}
        for agent in self.agents:
            base = obs_dict[agent].astype(np.float32)
            msg_in = np.zeros((self.incoming_msg_dim,), dtype=np.float32)
            obs_aug[agent] = np.concatenate([base, msg_in], axis=0)
        return obs_aug, infos

    def _sanitize_action(self, agent, act):
        space = self._env.action_space(agent)
        arr = np.array(act, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        low, high = space.low, space.high
        arr = np.clip(arr, low, high)
        arr = arr.reshape(space.shape)
        return arr.astype(np.float32)

    def _to_one_hot(self, idx_or_vec):
        # accept int or one-hot vector
        if isinstance(idx_or_vec, (int, np.integer)):
            oh = np.zeros((self.K_vocab,), dtype=np.float32)
            if 0 <= int(idx_or_vec) < self.K_vocab:
                oh[int(idx_or_vec)] = 1.0
            return oh
        vec = np.array(idx_or_vec, dtype=np.float32).reshape(-1)
        if vec.size == self.K_vocab:
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
            oh = np.zeros((self.K_vocab,), dtype=np.float32)
            oh[int(np.argmax(vec))] = 1.0
            return oh
        return np.zeros((self.K_vocab,), dtype=np.float32)

    def _build_obs_with_msgs(self, base_obs_dict, comm_on):
        obs_aug = {}
        for agent in self.agents:
            base = base_obs_dict[agent].astype(np.float32)
            msgs = []
            for other in self.agents:
                if other == agent:
                    continue
                msgs.append(self._last_msgs.get(other, np.zeros((self.K_vocab,), dtype=np.float32)))
            msg_in = np.concatenate(msgs, axis=0) if msgs else np.zeros((0,), dtype=np.float32)
            if not comm_on:
                msg_in = np.zeros_like(msg_in, dtype=np.float32)
            obs_aug[agent] = np.concatenate([base, msg_in], axis=0)
        return obs_aug

    def step(self, actions_dict, msg_dict_or_none, comm_on: bool):
        # sanitize actions
        act_dict_clean = {}
        for agent in self.agents:
            act = actions_dict.get(agent, None)
            if act is None:
                zeros = np.zeros(self._env.action_space(agent).shape, dtype=np.float32)
                act_dict_clean[agent] = self._sanitize_action(agent, zeros)
            else:
                act_dict_clean[agent] = self._sanitize_action(agent, act)

        # update messages if comm ON
        if comm_on and (msg_dict_or_none is not None):
            for agent in self.agents:
                m = msg_dict_or_none.get(agent, 0)
                self._last_msgs[agent] = self._to_one_hot(m)

        # step base env
        step_out = self._env.step(act_dict_clean)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs_next_base, rewards, terminations, truncations, infos = step_out
            dones = {a: bool(terminations.get(a, False) or truncations.get(a, False)) for a in self.agents}
        else:
            obs_next_base, rewards, dones, infos = step_out

        # ---------- Reward Shaping ----------
        shaped_rewards = {}
        avg_dist = self.avg_pairwise_distance_from_obs(obs_next_base)

        for agent in self.agents:
            r = rewards[agent]

            # Encourage spreading
            if avg_dist > 1.0:
                r += 0.05

            # Penalize collisions if info available
            if "collision" in infos.get(agent, {}):
                if infos[agent]["collision"]:
                    r -= 0.1

            # Encourage landmark coverage (if positions in info)
            if "landmark_dist" in infos.get(agent, {}):
                if infos[agent]["landmark_dist"] < 0.1:
                    r += 1.0

            shaped_rewards[agent] = r

        rewards = shaped_rewards
        # ------------------------------------

        # build augmented obs (t+1 incoming msgs)
        obs_next_aug = self._build_obs_with_msgs(obs_next_base, comm_on=comm_on)
        return obs_next_aug, rewards, dones, infos

    def get_state(self, obs_with_msgs_dict):
        base_all = []
        for agent in self.agents:
            obs = obs_with_msgs_dict[agent]
            base = obs[: self.base_obs_dim].astype(np.float32)
            base_all.append(base)
        msgs_all = [self._last_msgs[agent].astype(np.float32) for agent in self.agents]
        return np.concatenate(base_all + msgs_all, axis=0)

    def infer_other_relpos_slice(self, obs_vec):
        obs_dim = self.base_obs_dim
        N = self.num_agents
        tail_other = 2 * (N - 1)
        start = obs_dim - tail_other
        end = obs_dim
        return start, end

    def avg_pairwise_distance_from_obs(self, obs_with_msgs_dict):
        N = self.num_agents
        base = self.base_obs_dim
        L_est = (base - 4 - 4 * (N - 1)) / 2.0
        L = max(0, int(round(L_est)))

        start = 4 + 2 * L
        end = start + 2 * (N - 1)
        start = max(0, min(start, base))
        end = max(start, min(end, base))

        dists = []
        for agent in self.agents:
            obs = obs_with_msgs_dict[agent]
            rel = obs[:base][start:end]
            if rel.size:
                rel = rel.reshape(-1, 2)
                dists.extend(np.linalg.norm(rel, axis=1).tolist())
        return float(np.mean(dists)) if dists else 0.0

    # expose spaces
    def observation_space(self, agent):
        from gymnasium.spaces import Box
        return Box(low=-np.inf, high=np.inf, shape=(self.total_obs_dim,), dtype=np.float32)

    def action_space(self, agent):
        return self._env.action_space(agent)
