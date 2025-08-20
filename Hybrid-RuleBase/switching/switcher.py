import numpy as np
from collections import deque

class RuleBasedSwitcher:
    def __init__(self, entropy_threshold=0.4, overhead_threshold=0.5,
                 plateau_window=20, plateau_slope=1e-3,
                 entropy_window=5, hold_k=5):
        """
        Args:
            entropy_threshold: if avg entropy > threshold → trigger comm
            overhead_threshold: max comm usage allowed (fraction of steps)
            plateau_window: number of past episodes to detect reward plateau
            plateau_slope: slope threshold for plateau detection
            entropy_window: smoothing window for entropy moving average
            hold_k: hysteresis → keep comm ON for at least k steps once triggered
        """
        self.entropy_threshold = entropy_threshold
        self.overhead_threshold = overhead_threshold
        self.plateau_window = plateau_window
        self.plateau_slope = plateau_slope
        self.entropy_window = entropy_window
        self.hold_k = hold_k

        # histories
        self.reward_history = deque(maxlen=plateau_window)
        self.entropy_history = deque(maxlen=entropy_window)

        # hysteresis counter
        self.comm_hold_counter = 0

    def compute_entropy(self, logits: np.ndarray) -> float:
        """Shannon entropy from logits."""
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        return -np.sum(probs * np.log(probs + 1e-8))

    def reward_plateau_detected(self) -> bool:
        """Detect reward stagnation via slope of moving average."""
        if len(self.reward_history) < self.plateau_window:
            return False
        y = np.array(self.reward_history)
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        return abs(slope) < self.plateau_slope

    def update(self, obs_dict, action_logits_dict, reward_dict, comm_overhead: float):
        """
        Decide comm_on (True for DIAL, False for CTDE).
        Args:
            obs_dict: dict of observations (unused directly here)
            action_logits_dict: dict of agent logits
            reward_dict: dict of agent rewards
            comm_overhead: current comm usage ratio in episode
        """
        # === Step 1: Entropy check (smoothed) ===
        entropies = [self.compute_entropy(logits) for logits in action_logits_dict.values()]
        mean_entropy = float(np.mean(entropies))
        self.entropy_history.append(mean_entropy)
        smoothed_entropy = np.mean(self.entropy_history)

        # === Step 2: Reward plateau check ===
        mean_reward = np.mean(list(reward_dict.values()))
        self.reward_history.append(mean_reward)
        plateau = self.reward_plateau_detected()

        # === Step 3: Decision with hysteresis ===
        if self.comm_hold_counter > 0:
            # still holding comm ON
            self.comm_hold_counter -= 1
            return True

        if smoothed_entropy > self.entropy_threshold and plateau:
            if comm_overhead < self.overhead_threshold:
                self.comm_hold_counter = self.hold_k  # lock comm ON for k steps
                return True

        # default: comm OFF
        return False
