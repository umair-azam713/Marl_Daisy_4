# train_ctde_simple_spread.py
import os
import numpy as np
import torch
import torch.nn.functional as F

from envs.simple_spread_env import SimpleSpreadCTDE
from models.comm_policy import CommPolicy
from models.central_critic import CentralCritic
from utils.csv_logger import CSVLogger

# ---------------- Config ----------------
SEED = 1
NUM_EPISODES = 1000
MAX_STEPS = 50
K_VOCAB = 4
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
GAMMA = 0.95
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
UPDATE_EPOCHS = 4
MINIBATCHES = 4
ENTROPY_COEF = 0.01
MSG_ENTROPY_COEF = 0.01
VAL_COEF = 0.5
DEVICE = "cpu"

RESULTS_DIR = "results_ctde"
CSV_PATH = os.path.join(RESULTS_DIR, "ctde_simple_spread.csv")


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


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(SEED)

    # Env (PARALLEL API in our wrapper)
    env = SimpleSpreadCTDE(num_agents=3, max_cycles=MAX_STEPS, continuous_actions=True, K_vocab=K_VOCAB)
    obs_dict = env.reset(seed=SEED)
    agents = list(obs_dict.keys())
    num_agents = len(agents)

    # Dims
    obs_dim = len(next(iter(obs_dict.values())))
    act_dim = env.action_space(agents[0]).shape[0]
    a_low = env.action_space(agents[0]).low.astype(np.float32)
    a_high = env.action_space(agents[0]).high.astype(np.float32)

    # Actors and centralized critic
    actors = [
        CommPolicy(obs_dim, act_dim, K_vocab=K_VOCAB,
                   action_low=float(a_low[0]), action_high=float(a_high[0])).to(DEVICE)
        for _ in range(num_agents)
    ]
    critic = CentralCritic(state_dim=num_agents * obs_dim).to(DEVICE)

    actor_opts = [torch.optim.Adam(a.parameters(), lr=LR_ACTOR) for a in actors]
    critic_opt = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)

    # Logger
    fieldnames = [
        "episode", "return_sum", "return_mean",
        "reward_a0", "reward_a1", "reward_a2",
        "steps", "msgs_per_ep", "msg_entropy_ep", "seed", "K_vocab"
    ]
    csv = CSVLogger(CSV_PATH, fieldnames)

    # -------- Training --------
    for ep in range(1, NUM_EPISODES + 1):
        obs_dict = env.reset(seed=SEED + ep)
        ep_rewards = {ag: 0.0 for ag in agents}
        ep_msg_entropies = []
        steps = 0
        msgs_sent = 0

        # Per-agent buffers
        buf_obs = {ag: [] for ag in agents}
        buf_act = {ag: [] for ag in agents}
        buf_msg_idx = {ag: [] for ag in agents}
        buf_logp_a = {ag: [] for ag in agents}
        buf_logp_m = {ag: [] for ag in agents}
        buf_val = {ag: [] for ag in agents}
        buf_rew = {ag: [] for ag in agents}
        buf_done = {ag: [] for ag in agents}

        # Also store global states (same order as time)
        global_states = []

        while steps < MAX_STEPS:
            actions_dict = {}
            messages_dict = {}

            current_agents = list(env.env.agents)

            # Build global state BEFORE stepping (critic targets align with obs at t)
            gs = np.concatenate([obs_dict[ag].astype(np.float32) for ag in current_agents], axis=0)
            global_states.append(gs)
            val = critic(to_tensor(gs).unsqueeze(0)).item()

            for i, ag in enumerate(current_agents):
                o = to_tensor(obs_dict[ag]).unsqueeze(0)
                out = actors[i].act(o)

                action = out["action"].squeeze(0)
                logp_a = out["action_logp"].squeeze(0).item()
                msg_idx = int(out["msg_idx"].item())
                logp_m = out["msg_logp"].squeeze(0).item()
                msg_logits = out["msg_logits"].squeeze(0)

                # Sanitize + shape action for env
                a_np = action.detach().cpu().numpy().astype(np.float32)
                a_np = np.nan_to_num(a_np, nan=0.0, posinf=0.0, neginf=0.0)
                a_np = np.clip(a_np, a_low, a_high).astype(np.float32)
                a_np = np.array(a_np, dtype=np.float32).reshape(env.action_space(ag).shape)

                actions_dict[ag] = a_np
                messages_dict[ag] = msg_idx  # our env accepts int idx or one-hot

                # Store rollouts
                buf_obs[ag].append(obs_dict[ag].copy())
                buf_act[ag].append(a_np.copy())
                buf_msg_idx[ag].append(msg_idx)
                buf_logp_a[ag].append(logp_a)
                buf_logp_m[ag].append(logp_m)
                buf_val[ag].append(val)   # same centralized value for all agents at t
                buf_done[ag].append(False)

                # Log message entropy (for monitoring)
                msg_probs = F.softmax(msg_logits, dim=-1)
                msg_ent = -(msg_probs * (msg_probs.clamp_min(1e-8).log())).sum().item()
                ep_msg_entropies.append(msg_ent)

            # Step env with both actions and messages
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

        # -------- PPO update (per agent) --------
        global_states = np.array(global_states, dtype=np.float32)  # [T, state_dim]

        for i, ag in enumerate(agents):
            obs_arr = np.array(buf_obs[ag], dtype=np.float32)        # [T, obs_dim]
            act_arr = np.array(buf_act[ag], dtype=np.float32)        # [T, act_dim]
            msg_idx_arr = np.array(buf_msg_idx[ag], dtype=np.int64)  # [T]
            logp_a_arr = np.array(buf_logp_a[ag], dtype=np.float32)  # [T]
            logp_m_arr = np.array(buf_logp_m[ag], dtype=np.float32)  # [T]
            val_arr = np.array(buf_val[ag] + [0.0], dtype=np.float32)  # [T+1]
            rew_arr = np.array(buf_rew[ag], dtype=np.float32)        # [T]
            done_arr = np.array(buf_done[ag], dtype=np.bool_)        # [T]

            adv, ret = compute_gae(rew_arr, done_arr, val_arr, gamma=GAMMA, lam=GAE_LAMBDA)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            T = len(rew_arr)
            idx_all = np.arange(T)
            mb_size = max(1, T // MINIBATCHES)

            for _ in range(UPDATE_EPOCHS):
                np.random.shuffle(idx_all)
                for start in range(0, T, mb_size):
                    idx = idx_all[start:start + mb_size]

                    o = to_tensor(obs_arr[idx])
                    a = to_tensor(act_arr[idx])
                    m_idx = torch.as_tensor(msg_idx_arr[idx], dtype=torch.long, device=DEVICE)
                    old_logp_a = to_tensor(logp_a_arr[idx])
                    old_logp_m = to_tensor(logp_m_arr[idx])
                    adv_b = to_tensor(adv[idx])
                    ret_b = to_tensor(ret[idx])

                    # Actor evaluate (needs msg_idx)
                    ev = actors[i].evaluate_actions(o, a, m_idx)
                    # Keys from CommPolicy.evaluate_actions:
                    #   "logp_a", "entropy_a", "logp_m", "entropy_m"
                    ratio_a = torch.exp(ev["logp_a"] - old_logp_a)
                    ratio_m = torch.exp(ev["logp_m"] - old_logp_m)

                    clip_a = torch.clamp(ratio_a, 1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO) * adv_b
                    clip_m = torch.clamp(ratio_m, 1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO) * adv_b

                    policy_loss_a = -(torch.min(ratio_a * adv_b, clip_a)).mean()
                    policy_loss_m = -(torch.min(ratio_m * adv_b, clip_m)).mean()

                    entropy_a = ev["entropy_a"].mean()
                    entropy_m = ev["entropy_m"].mean()

                    # Critic on the corresponding global states
                    state_b = to_tensor(global_states[idx])  # [B, state_dim]
                    value_pred = critic(state_b)
                    value_loss = F.mse_loss(value_pred, ret_b)

                    loss = policy_loss_a + policy_loss_m \
                           + VAL_COEF * value_loss \
                           - ENTROPY_COEF * entropy_a \
                           - MSG_ENTROPY_COEF * entropy_m

                    actor_opts[i].zero_grad()
                    critic_opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actors[i].parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    actor_opts[i].step()
                    critic_opt.step()

        # -------- Logging --------
        returns = [ep_rewards[ag] for ag in agents]
        row = {
            "episode": ep,
            "return_sum": float(np.sum(returns)),
            "return_mean": float(np.mean(returns)),
            "reward_a0": returns[0],
            "reward_a1": returns[1],
            "reward_a2": returns[2],
            "steps": steps,
            "msgs_per_ep": msgs_sent,
            "msg_entropy_ep": safe_mean(ep_msg_entropies),
            "seed": SEED,
            "K_vocab": K_VOCAB,
        }
        csv.log(row)

        if ep % 50 == 0 or ep == 1:
            print(f"[EP {ep:4d}] return_mean={row['return_mean']:.3f} "
                  f"steps={steps} msgs={msgs_sent} "
                  f"msgH={row['msg_entropy_ep']:.3f}")

    print(f"Training complete. CSV saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()
