# envs/simple_spread_env.py
import numpy as np
from gymnasium import spaces
from pettingzoo.mpe import simple_spread_v3

class SimpleSpreadWithComm:
    """
    Parallel wrapper for simple_spread_v3 with explicit DIAL-style communication.
    Messages sent at t are received by other agents at t+1.
    """
    def __init__(self, num_agents=3, max_cycles=50, continuous_actions=True, K_vocab=4):
        self.num_agents = num_agents
        self.K_vocab = K_vocab
        self.env = simple_spread_v3.parallel_env(
            N=num_agents,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions
        )
        # messages are stored in a fixed agent order (self.env.agents)
        self.last_messages = [np.zeros(K_vocab, dtype=np.float32) for _ in range(num_agents)]

    def reset(self, seed=None):
        # Gymnasium-style API returns (obs_dict, infos)
        obs_dict, _infos = self.env.reset(seed=seed)
        self.last_messages = [np.zeros(self.K_vocab, dtype=np.float32) for _ in range(self.num_agents)]
        return self._concat_messages(obs_dict)

    # envs/simple_spread_env.py  (only the step() method changes)
    def step(self, actions_dict, messages_dict):
        """
        actions_dict: dict(agent_name -> np.array action)
        messages_dict: dict(agent_name -> one-hot message vector)
        """
        # Ensure we provide an action for EVERY agent the env expects
        actions_complete = {}
        for a in self.env.agents:
            if a in actions_dict:
                act = actions_dict[a]
            else:
                # fallback: zero action with correct shape/dtype
                sample = self.env.action_space(a).sample()
                act = np.zeros_like(sample, dtype=np.float32)
            actions_complete[a] = act

        # Step the env
        obs_dict, rewards, terminations, truncations, infos = self.env.step(actions_complete)

        # Combine terminations and truncations into dones
        dones = {a: bool(terminations.get(a, False) or truncations.get(a, False)) for a in self.env.agents}

        # Store messages sent at current step; default to zero if missing
        new_msgs = []
        for a in self.env.agents:
            if a in messages_dict:
                new_msgs.append(messages_dict[a])
            else:
                new_msgs.append(np.zeros(self.K_vocab, dtype=np.float32))
        self.last_messages = new_msgs

        obs_with_msgs = self._concat_messages(obs_dict)
        return obs_with_msgs, rewards, dones, infos


    def _concat_messages(self, obs_dict):
        """
        Concatenate base observation with incoming messages (from other agents).
        """
        obs_with_msgs = {}
        agents = self.env.agents
        for i, agent in enumerate(agents):
            incoming = np.concatenate([self.last_messages[j] for j in range(self.num_agents) if j != i], dtype=np.float32)
            obs_with_msgs[agent] = np.concatenate([obs_dict[agent], incoming], dtype=np.float32)
        return obs_with_msgs

    def observation_space(self, agent):
        base_dim = self.env.observation_space(agent).shape[0]
        comm_dim = (self.num_agents - 1) * self.K_vocab
        return spaces.Box(low=-np.inf, high=np.inf, shape=(base_dim + comm_dim,), dtype=np.float32)

    def action_space(self, agent):
        return self.env.action_space(agent)
