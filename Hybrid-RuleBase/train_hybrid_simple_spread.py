# HYBRID/train_baseline_simple_spread.py
import os
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict

from envs.simple_spread_env_hybrid import SimpleSpreadHybridWrapper
from models.comm_policy import CommPolicy
from models.central_critic import CentralCritic
from switching.switcher import RuleBasedSwitcher
from utils.misc import set_seed, to_tensor, compute_gae, safe_mean
from utils.csv_logger import CSVLogger


def ppo_update(agent_name, policy, optimizer, obs, acts, msg_idx, old_logp_a, old_logp_m, advantages,
               returns, critic, states, clip_ratio, epochs, minibatches,
               entropy_coef, msg_entropy_coef, val_coef, device):
    N = obs.shape[0]
    idxs = np.arange(N)
    for _ in range(epochs):
        np.random.shuffle(idxs)
        mb_size = max(1, N // minibatches)
        for start in range(0, N, mb_size):
            end = start + mb_size
            mb = idxs[start:end]
            obs_b = to_tensor(obs[mb], device)
            acts_b = to_tensor(acts[mb], device)
            msg_b = torch.from_numpy(msg_idx[mb]).long().to(device)
            adv_b = to_tensor(advantages[mb], device)
            ret_b = to_tensor(returns[mb], device)
            old_logp_a_b = to_tensor(old_logp_a[mb], device)
            old_logp_m_b = to_tensor(old_logp_m[mb], device)
            states_b = to_tensor(states[mb], device)

            out = policy.evaluate_actions(obs_b, acts_b, msg_b)
            logp_a, ent_a = out['logp_a'], out['entropy_a']
            logp_m, ent_m = out['logp_m'], out['entropy_m']

            ratio_a = torch.exp(logp_a - old_logp_a_b)
            ratio_m = torch.exp(logp_m - old_logp_m_b)

            surr1_a = ratio_a * adv_b
            surr2_a = torch.clamp(ratio_a, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_b
            policy_loss_a = -torch.min(surr1_a, surr2_a).mean()

            surr1_m = ratio_m * adv_b
            surr2_m = torch.clamp(ratio_m, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_b
            policy_loss_m = -torch.min(surr1_m, surr2_m).mean()

            # Central critic value loss on global states
            values = critic(states_b)
            value_loss = ((values - ret_b) ** 2).mean()

            entropy_bonus = entropy_coef * ent_a.mean() + msg_entropy_coef * ent_m.mean()
            loss = policy_loss_a + policy_loss_m + val_coef * value_loss - entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
    ...


def main(mode="hybrid"):
    """
    mode: "hybrid" | "ctde" | "dial" | "random"
    """
    # Defaults
    episodes = 1000
    steps = 100
    K_vocab = 4
    lr = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_ratio = 0.2
    update_epochs = 5
    minibatches = 4
    entropy_coef = 0.01
    msg_entropy_coef = 0.2
    val_coef = 0.5
    seed = 42

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    # Env
    env = SimpleSpreadHybridWrapper(N=3, K_vocab=K_vocab, max_cycles=steps, seed=seed)
    agents = env.agents

    # Policies per agent
    obs_dim = env.total_obs_dim
    act_dim = int(np.prod(env.action_space(agents[0]).shape))
    policies = {a: CommPolicy(obs_dim, act_dim, K_vocab=K_vocab).to(device) for a in agents}
    optims = {a: optim.Adam(policies[a].parameters(), lr=lr) for a in agents}

    # Central critic
    state_dim = env.base_obs_dim * len(agents) + K_vocab * len(agents)
    critic = CentralCritic(state_dim).to(device)
    critic_opt = optim.Adam(critic.parameters(), lr=lr)

    # Switcher for hybrid only
    switcher = RuleBasedSwitcher(
        entropy_threshold=0.3,
        overhead_threshold=0.5,
        plateau_window=100,
        plateau_slope=1e-4,
        entropy_window=100,
        hold_k=100
    )

    # === Output folder per mode ===
    out_dir = f"results_{mode}"
    os.makedirs(out_dir, exist_ok=True)
    logger = CSVLogger(os.path.join(out_dir, 'metrics.csv'),
                       fieldnames=['episode','return_sum','return_mean',
                                   'reward_a0','reward_a1','reward_a2',
                                   'steps','msgs_per_ep','msg_entropy_ep',
                                   'pct_comm_on','avg_pair_dist'])

    for ep in range(1, episodes + 1):
        obs_dict, infos = env.reset(seed=seed + ep)

        traj = {a: defaultdict(list) for a in agents}
        states_list = []
        ep_rewards = {a: 0.0 for a in agents}
        msgs_count = 0
        msg_entropy_acc = []
        comm_on_count = 0
        avg_pair_dists = []

        for t in range(steps):
            comm_overhead = msgs_count / (1 + t * len(agents))
            reward_dict = {a: 0.0 for a in agents} if t == 0 else rewards

            actions_dict, messages_dict = {}, {}
            msg_logits_ep, action_logits_dict = [], {}

            for a in agents:
                o = obs_dict[a].astype(np.float32)
                o_t = to_tensor(o, device)
                act, logp_a, m_idx, logp_m, msg_logits = policies[a].act(o_t)
                actions_dict[a] = act.cpu().numpy()
                action_logits_dict[a] = msg_logits.detach().cpu().numpy()

                traj[a]['obs'].append(o)
                traj[a]['act'].append(act.cpu().numpy())
                traj[a]['msg_idx'].append(int(m_idx.item()))
                traj[a]['logp_a'].append(float(logp_a.item()))
                traj[a]['logp_m'].append(float(logp_m.item()))

                msg_logits_ep.append(msg_logits.detach().cpu().numpy())

            # === Communication decision ===
            if mode == "hybrid":
                comm_on = switcher.update(obs_dict, action_logits_dict, reward_dict, comm_overhead)
            elif mode == "dial":
                comm_on = True
            elif mode == "ctde":
                comm_on = False
            elif mode == "random":
                comm_on = np.random.rand() < 0.5
            else:
                raise ValueError(f"Unknown mode {mode}")

            if comm_on:
                comm_on_count += 1
                for a in agents:
                    messages_dict[a] = traj[a]['msg_idx'][-1]

            state = env.get_state(obs_dict)
            states_list.append(state)

            obs_next, rewards, dones, infos = env.step(
                actions_dict, messages_dict if comm_on else None, comm_on
            )

            for a in agents:
                r = float(rewards[a])
                traj[a]['rew'].append(r)
                ep_rewards[a] += r
                traj[a]['done'].append(bool(dones[a]))
            if comm_on:
                msgs_count += len(agents)
                for a in agents:
                    traj[a]['rew'][-1] -= 0.01

            # entropy
            logits_stack = np.stack(msg_logits_ep, axis=0)
            probs = np.exp(logits_stack - logits_stack.max(axis=1, keepdims=True))
            probs /= np.clip(probs.sum(axis=1, keepdims=True), 1e-8, None)
            entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1).mean()
            msg_entropy_acc.append(float(entropy))

            avg_pair_dists.append(env.avg_pairwise_distance_from_obs(obs_dict))
            obs_dict = obs_next

        # --- End of episode logging ---
        pct_comm_on = comm_on_count / steps
        avg_pair_dist = safe_mean(avg_pair_dists)

        row = {
            'episode': ep,
            'return_sum': sum(ep_rewards.values()),
            'return_mean': safe_mean(list(ep_rewards.values())),
            'reward_a0': ep_rewards[agents[0]],
            'reward_a1': ep_rewards[agents[1]],
            'reward_a2': ep_rewards[agents[2]],
            'steps': steps,
            'msgs_per_ep': msgs_count,
            'msg_entropy_ep': safe_mean(msg_entropy_acc),
            'pct_comm_on': pct_comm_on,
            'avg_pair_dist': avg_pair_dist,
        }
        logger.log(row)

        if ep % 10 == 0:
            print(f"[{mode}] Ep {ep}: return_mean={row['return_mean']:.3f} "
                  f"pct_comm_on={pct_comm_on:.2f} msgs={msgs_count}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["hybrid", "ctde", "dial", "random"])
    args = parser.parse_args()
    main(mode=args.mode)
