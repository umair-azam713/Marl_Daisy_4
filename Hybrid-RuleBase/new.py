# # HYBRID/analyze_hybrid.py
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# def load_results(csv_path, label, color):
#     """Helper to load one CSV run with consistent column names."""
#     cols = ['episode','return_sum','return_mean',
#             'reward_a0','reward_a1','reward_a2',
#             'steps','msgs_per_ep','msg_entropy_ep',
#             'pct_comm_on','avg_pair_dist']
#     df = pd.read_csv(csv_path, names=cols)
#     df["label"] = label
#     df["color"] = color
#     return df

# def analyze_results(hybrid_csv="results_hybrid/metrics.csv",
#                     dial_csv=None, ctde_csv=None, random_csv=None,
#                     out_dir="results_hybrid/plots",
#                     use_dummy_baselines=True):
#     os.makedirs(out_dir, exist_ok=True)

#     # === Load Hybrid run ===
#     hybrid_df = load_results(hybrid_csv, "Hybrid (Switching)", "orange")

#     dfs = [hybrid_df]

#     # === Optionally load real baselines ===
#     if dial_csv: dfs.append(load_results(dial_csv, "DIAL-only", "red"))
#     if ctde_csv: dfs.append(load_results(ctde_csv, "CTDE-only", "blue"))
#     if random_csv: dfs.append(load_results(random_csv, "Random Switching", "purple"))

#     # === Or add dummy baselines (for paper-style figs) ===
#     if use_dummy_baselines and len(dfs) == 1:
#         episodes = hybrid_df["episode"]
#         steps = int(hybrid_df["steps"].iloc[0]) 

#         # DIAL-only: higher rewards than CTDE, constant high comm
#         dial_df = pd.DataFrame({
#             "episode": episodes,
#             "return_mean": np.linspace(-200, -50, len(episodes)) + np.random.normal(0, 5, len(episodes)),
#             "msgs_per_ep": np.full(len(episodes), steps),  # all agents always talking
#             "steps": steps,
#             "label": "DIAL-only",
#             "color": "red"
#         })
#         dfs.append(dial_df)

#         # CTDE-only: lower rewards, zero comm
#         ctde_df = pd.DataFrame({
#             "episode": episodes,
#             "return_mean": np.linspace(-250, -100, len(episodes)) + np.random.normal(0, 5, len(episodes)),
#             "msgs_per_ep": np.zeros(len(episodes)),
#             "steps": steps,
#             "label": "CTDE-only",
#             "color": "blue"
#         })
#         dfs.append(ctde_df)

#         # Random switching: sits between
#         rand_df = pd.DataFrame({
#             "episode": episodes,
#             "return_mean": np.linspace(-230, -80, len(episodes)) + np.random.normal(0, 8, len(episodes)),
#             "msgs_per_ep": np.random.randint(0, steps//2, len(episodes)),
#             "steps": steps,
#             "label": "Random Switching",
#             "color": "purple"
#         })
#         dfs.append(rand_df)

#     all_data = pd.concat(dfs, ignore_index=True)

#     # === Figure 1: Learning Curve Comparison (paper style) ===
#     plt.figure(figsize=(8,5))
#     for label, group in all_data.groupby("label"):
#         plt.plot(group["episode"],
#                  group["return_mean"].rolling(50).mean(),
#                  label=label, color=group["color"].iloc[0], linewidth=2)
#     plt.xlabel("Episodes")
#     plt.ylabel("Cumulative Reward")
#     plt.title("Learning Curves")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(out_dir, "learning_curve_baselines.png"))
#     plt.close()

#     # === Figure 2: Communication Overhead Comparison (paper style) ===
#     plt.figure(figsize=(8,5))
#     for label, group in all_data.groupby("label"):
#         overhead = group["msgs_per_ep"] / group["steps"]
#         plt.plot(group["episode"],
#                  overhead.rolling(50).mean(),
#                  label=label, color=group["color"].iloc[0], linewidth=2)
#     plt.xlabel("Episodes")
#     plt.ylabel("Messages per Agent per Step")
#     plt.title("Communication Overhead")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(out_dir, "comm_overhead_baselines.png"))
#     plt.close()

#     print("✅ Paper-style baseline plots saved in:", out_dir)


# if __name__ == "__main__":
#     # Run with Hybrid only; dummy baselines will be added
#     analyze_results(
#         hybrid_csv="results_hybrid/metrics.csv",
#         dial_csv=None,      # supply real CSV if available
#         ctde_csv=None,
#         random_csv=None,
#         use_dummy_baselines=True
#     )

import os
import pandas as pd
import matplotlib.pyplot as plt

def load_results(csv_path, label, color):
    """Helper to load one CSV run with consistent column names."""
    df = pd.read_csv(csv_path)  # <-- FIX: read header row properly
    df["label"] = label
    df["color"] = color
    return df

def analyze_results(hybrid_csv="results_hybrid/metrics.csv",
                    dial_csv=None, ctde_csv=None, random_csv=None,
                    out_dir="results_plots_1"):
    os.makedirs(out_dir, exist_ok=True)

    # === Load Hybrid ===
    hybrid_df = load_results(hybrid_csv, "Hybrid (Switching)", "orange")
    print("\n📊 Preview: Hybrid (Switching)")
    print(hybrid_df[["episode", "return_mean"]].head())

    # --- Learning curve (return_mean over episodes) ---
    plt.figure(figsize=(8, 5))
    plt.plot(hybrid_df["episode"], hybrid_df["return_mean"], alpha=0.5, label="Raw")
    plt.plot(hybrid_df["episode"],
             hybrid_df["return_mean"].rolling(50).mean(),
             label="Moving Avg (50)", linewidth=2, color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Mean Return")
    plt.title("Learning Curve (Hybrid Only)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "learning_curve.png"))
    plt.close()

    # --- Communication usage (% comm ON) ---
    plt.figure(figsize=(8, 5))
    plt.plot(hybrid_df["episode"], hybrid_df["pct_comm_on"], color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Fraction of Steps with Comm")
    plt.title("Communication Switching (CTDE ↔ DIAL)")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "comm_usage.png"))
    plt.close()

    # --- Messages per episode ---
    plt.figure(figsize=(8, 5))
    plt.plot(hybrid_df["episode"], hybrid_df["msgs_per_ep"], color="green")
    plt.xlabel("Episode")
    plt.ylabel("Total Messages Sent")
    plt.title("Communication Load")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "msgs_per_ep.png"))
    plt.close()

    # --- Message entropy ---
    plt.figure(figsize=(8, 5))
    plt.plot(hybrid_df["episode"], hybrid_df["msg_entropy_ep"], color="red")
    plt.axhline(y=1.386, linestyle="--", color="black", label="Uniform(4) Entropy")
    plt.xlabel("Episode")
    plt.ylabel("Entropy")
    plt.title("Message Distribution Entropy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "msg_entropy.png"))
    plt.close()

    # --- Avg pairwise distance ---
    plt.figure(figsize=(8, 5))
    plt.plot(hybrid_df["episode"], hybrid_df["avg_pair_dist"], color="purple")
    plt.xlabel("Episode")
    plt.ylabel("Distance")
    plt.title("Agent Spreading Behavior")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "avg_pair_dist.png"))
    plt.close()

    # --- First vs Last 100 episodes ---
    first100 = hybrid_df[hybrid_df["episode"] <= 100].mean(numeric_only=True)
    last100 = hybrid_df[hybrid_df["episode"] > hybrid_df["episode"].max() - 100].mean(numeric_only=True)

    metrics = ["return_mean", "msgs_per_ep", "pct_comm_on", "msg_entropy_ep", "avg_pair_dist"]
    plt.figure(figsize=(10, 6))
    x = range(len(metrics))
    plt.bar([i - 0.2 for i in x], [first100[m] for m in metrics], width=0.4, label="First 100")
    plt.bar([i + 0.2 for i in x], [last100[m] for m in metrics], width=0.4, label="Last 100")
    plt.xticks(x, metrics, rotation=30)
    plt.title("First vs Last 100 Episodes")
    plt.legend()
    plt.grid(True, axis="y")
    plt.savefig(os.path.join(out_dir, "first_vs_last.png"))
    plt.close()

    print("✅ Hybrid-only analysis done. Plots saved in:", out_dir)

    # === Baseline Comparisons (if CSVs provided) ===
    dfs = [hybrid_df]
    if dial_csv:
        dial_df = load_results(dial_csv, "DIAL-only", "red")
        print("\n📊 Preview: DIAL-only")
        print(dial_df[["episode", "return_mean"]].head())
        dfs.append(dial_df)
    if ctde_csv:
        ctde_df = load_results(ctde_csv, "CTDE-only", "blue")
        print("\n📊 Preview: CTDE-only")
        print(ctde_df[["episode", "return_mean"]].head())
        dfs.append(ctde_df)
    if random_csv:
        random_df = load_results(random_csv, "Random Switching", "purple")
        print("\n📊 Preview: Random Switching")
        print(random_df[["episode", "return_mean"]].head())
        dfs.append(random_df)

    if len(dfs) > 1:
        all_data = pd.concat(dfs, ignore_index=True)

        # Learning Curves
        plt.figure(figsize=(8,5))
        for label, group in all_data.groupby("label"):
            plt.plot(group["episode"], group["return_mean"].rolling(50).mean(),
                     label=label, color=group["color"].iloc[0])
        plt.xlabel("Episodes")
        plt.ylabel("Mean Return")
        plt.title("Learning Curves (Baselines)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "learning_curve_baselines.png"))
        plt.close()

        # Communication Overhead
        plt.figure(figsize=(8,5))
        for label, group in all_data.groupby("label"):
            overhead = group["msgs_per_ep"] / group["steps"]
            plt.plot(group["episode"], overhead.rolling(50).mean(),
                     label=label, color=group["color"].iloc[0])
        plt.xlabel("Episodes")
        plt.ylabel("Messages per Agent per Step")
        plt.title("Communication Overhead (Baselines)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "comm_overhead_baselines.png"))
        plt.close()

        print("✅ Baseline comparison plots saved in:", out_dir)


if __name__ == "__main__":
    analyze_results(
        hybrid_csv="results_hybrid/metrics_updated_hybrid.csv",
        dial_csv="results_dial/dial_simple_spread.csv",
        ctde_csv="results_ctde/ctde_simple_spread_fixed.csv",
        
    )
