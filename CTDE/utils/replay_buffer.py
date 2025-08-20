# utils/replay_buffer.py
import numpy as np


class ReplayBuffer:
    def __init__(self):
        self.storage = {}

    def init_agent(self, agent_name):
        self.storage[agent_name] = {
            "obs": [],
            "actions": [],
            "msgs": [],
            "logp_a": [],
            "logp_m": [],
            "rewards": [],
            "values": [],
            "dones": []
        }

    def store(self, agent_name, obs, action, msg_idx, logp_a, logp_m, reward, value, done):
        buf = self.storage[agent_name]
        buf["obs"].append(obs)
        buf["actions"].append(action)
        buf["msgs"].append(msg_idx)
        buf["logp_a"].append(logp_a)
        buf["logp_m"].append(logp_m)
        buf["rewards"].append(reward)
        buf["values"].append(value)
        buf["dones"].append(done)

    def get_agent_data(self, agent_name):
        buf = self.storage[agent_name]
        return {k: np.array(v) for k, v in buf.items()}

    def clear(self):
        for ag in self.storage:
            for key in self.storage[ag]:
                self.storage[ag][key] = []
