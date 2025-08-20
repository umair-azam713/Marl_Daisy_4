# train_dial_simple_spread.py
import os
import math
import numpy as np
import torch
import torch.nn.functional as F

from envs.simple_spread_env import SimpleSpreadWithComm
from models.dial_policy import DIALPolicy
from utils.csv_logger import CSVLogger

# ---------------- Config ----------------
SEED = 1
NUM_EPISODES = 1000
MAX_STEPS = 50
K_VOCAB = 4
LR = 3e-4
GAMMA = 0.95
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
UPDATE_EPOCHS = 4
MINIBATCHES = 4
ENTROPY_COEF = 0.01         # action entropy
MSG_ENTROPY_COEF = 0.01     # message entropy
VAL_COEF = 0.5
DEVICE = "cpu"

RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "dial_simple_spread.csv")

# -------------- Helpers -----------------
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)

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

# -------------- Main --------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(SEED)

    # Env
    env = SimpleSpreadWithComm(num_agents=3, max_cycles=MAX_STEPS,
                               continuous_actions=True, K_vocab=K_VOCAB)
    obs_dict = env.reset(seed=SEED)
    agents = list(obs_dict.keys())
    num_agents = len(agents)

    # Obs/Act dims
    obs_dim = len(next(iter(obs_dict.values())))
    act_dim = env.action_space(agents[0]).shape[0]
    a_low = float(env.action_space(agents[0]).low[0])
    a_high = float(env.action_space(agents[0]).high[0])

    # Policies (independent PPO per agent)
    policies = []
    optimizers = []
    for _ in range(num_agents):
        pol = DIALPolicy(obs_dim, act_dim, K_vocab=K_VOCAB,
                         action_low=a_low, action_high=a_high,
                         msg_temp_init=1.0).to(DEVICE)
        policies.append(pol)
        optimizers.append(torch.optim.Adam(pol.parameters(), lr=LR))

    # CSV logger
    fieldnames = [
        "episode","return_sum","return_mean","reward_a0","reward_a1","reward_a2",
        "steps","msgs_per_ep","msg_entropy_ep","seed","K_vocab"
    ]
    csv_logger = CSVLogger(CSV_PATH, fieldnames)

    # Training loop
    for ep in range(1, NUM_EPISODES + 1):
        obs_dict = env.reset(seed=SEED + ep)
        ep_rewards = {ag: 0.0 for ag in agents}
        ep_msg_entropies = []
        steps = 0
        msgs_sent = 0

        # Rollout buffers (per-agent)
        buf_obs = {ag: [] for ag in agents}
        buf_act = {ag: [] for ag in agents}
        buf_logp = {ag: [] for ag in agents}
        buf_val = {ag: [] for ag in agents}
        buf_rew = {ag: [] for ag in agents}
        buf_done = {ag: [] for ag in agents}

        while steps < MAX_STEPS:
            actions_dict = {}
            messages_dict = {}

            # Always use current env.agents list to avoid KeyErrors
            current_agents = list(env.env.agents)

            for i, ag in enumerate(current_agents):
                o = to_tensor(obs_dict[ag]).unsqueeze(0)
                out = policies[i].act(o)

                action = out["action"].squeeze(0)
                logp   = out["action_logp"].squeeze(0)
                value  = out["value"].squeeze(0)
                msg_onehot = out["msg_onehot"].squeeze(0)
                msg_logits = out["msg_logits"].squeeze(0)

                # Safety: remove NaNs/Infs, clamp to bounds
                action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
                action = torch.clamp(action, min=a_low, max=a_high)

                # Store rollout data
                buf_obs[ag].append(obs_dict[ag].copy())
                buf_act[ag].append(action.detach().cpu().numpy())
                buf_logp[ag].append(logp.detach().cpu().item())
                buf_val[ag].append(value.detach().cpu().item())
                buf_done[ag].append(False)

                # Message entropy for logging
                msg_probs = F.softmax(msg_logits, dim=-1)
                msg_ent = -(msg_probs * (msg_probs.clamp_min(1e-8).log())).sum().item()
                ep_msg_entropies.append(msg_ent)

                actions_dict[ag] = action.detach().cpu().numpy()
                messages_dict[ag] = msg_onehot.detach().cpu().numpy()

            # Step env
            next_obs_dict, rewards, dones, infos = env.step(actions_dict, messages_dict)

            # Rewards
            for ag in current_agents:
                r = float(rewards[ag])
                ep_rewards[ag] += r
                buf_rew[ag].append(r)

            msgs_sent += len(current_agents)
            steps += 1
            obs_dict = next_obs_dict

            if all(dones.values()):
                break

        # -------- PPO Update --------
        for i, ag in enumerate(agents):
            obs_arr = np.array(buf_obs[ag], dtype=np.float32)
            act_arr = np.array(buf_act[ag], dtype=np.float32)
            logp_arr = np.array(buf_logp[ag], dtype=np.float32)
            val_arr = np.array(buf_val[ag] + [0.0], dtype=np.float32)
            rew_arr = np.array(buf_rew[ag], dtype=np.float32)
            done_arr = np.array(buf_done[ag], dtype=np.bool_)

            adv, ret = compute_gae(rew_arr, done_arr, val_arr,
                                   gamma=GAMMA, lam=GAE_LAMBDA)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            T = len(rew_arr)
            batch_idx = np.arange(T)
            mb_size = max(1, T // MINIBATCHES)

            for _ in range(UPDATE_EPOCHS):
                np.random.shuffle(batch_idx)
                for start in range(0, T, mb_size):
                    idx = batch_idx[start:start+mb_size]
                    o = to_tensor(obs_arr[idx])
                    a = to_tensor(act_arr[idx])
                    old_logp = to_tensor(logp_arr[idx])
                    adv_b = to_tensor(adv[idx])
                    ret_b = to_tensor(ret[idx])

                    ev = policies[i].evaluate_actions(o, a)
                    logp = ev["logp"]
                    entropy = ev["entropy"]
                    value = ev["value"]
                    msg_entropy = ev["msg_entropy"]

                    ratio = torch.exp(logp - old_logp)
                    clip_adv = torch.clamp(ratio, 1.0 - CLIP_RATIO,
                                           1.0 + CLIP_RATIO) * adv_b
                    policy_loss = -(torch.min(ratio * adv_b, clip_adv)).mean()
                    value_loss = F.mse_loss(value, ret_b)

                    loss = policy_loss \
                           + VAL_COEF * value_loss \
                           - ENTROPY_COEF * entropy.mean() \
                           - MSG_ENTROPY_COEF * msg_entropy.mean()

                    optimizers[i].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policies[i].parameters(), 1.0)
                    optimizers[i].step()

        # -------- Logging --------
        ret_vals = [ep_rewards[ag] for ag in agents]
        row = {
            "episode": ep,
            "return_sum": float(np.sum(ret_vals)),
            "return_mean": float(np.mean(ret_vals)),
            "reward_a0": ret_vals[0],
            "reward_a1": ret_vals[1],
            "reward_a2": ret_vals[2],
            "steps": steps,
            "msgs_per_ep": msgs_sent,
            "msg_entropy_ep": safe_mean(ep_msg_entropies),
            "seed": SEED,
            "K_vocab": K_VOCAB,
        }
        csv_logger.log(row)

        if ep % 50 == 0 or ep == 1:
            print(f"[EP {ep:4d}] return_mean={row['return_mean']:.3f} "
                  f"steps={steps} msgs={msgs_sent} "
                  f"msgH={row['msg_entropy_ep']:.3f}")

    print(f"Training complete. CSV saved to: {CSV_PATH}")

if __name__ == "__main__":
    main()
