# HYBRID/analyze_hybrid.py
import os
import pandas as pd
import matplotlib.pyplot as plt

def load_results(csv_path, label, color):
    """Helper to load one CSV run with consistent column names."""
    cols = ['episode','return_sum','return_mean',
            'reward_a0','reward_a1','reward_a2',
            'steps','msgs_per_ep','msg_entropy_ep',
            'pct_comm_on','avg_pair_dist']
    df = pd.read_csv(csv_path, names=cols)
    df["label"] = label
    df["color"] = color
    return df

def analyze_results(hybrid_csv="results_hybrid/metrics.csv",
                    dial_csv=None, ctde_csv=None, random_csv=None,
                    out_dir="results_hybrid/plots"):
    os.makedirs(out_dir, exist_ok=True)

    # === Load Hybrid (always) ===
    hybrid_df = load_results(hybrid_csv, "Hybrid (Switching)", "orange")

    # --- Learning curve (return_mean over episodes) ---
    plt.figure(figsize=(8, 5))
    plt.plot(hybrid_df["episode"], hybrid_df["return_mean"], label="Return Mean", alpha=0.7)
    hybrid_df["return_ma"] = hybrid_df["return_mean"].rolling(window=50).mean()
    plt.plot(hybrid_df["episode"], hybrid_df["return_ma"], label="Moving Avg (50)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Mean Return")
    plt.title("Learning Curve (Hybrid Only)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "learning_curve.png"))
    plt.close()

    # --- Communication usage (pct_comm_on per episode) ---
    plt.figure(figsize=(8, 5))
    plt.plot(hybrid_df["episode"], hybrid_df["pct_comm_on"], label="% Comm On (DIAL)", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Fraction of Steps with Comm")
    plt.title("Communication Switching (CTDE ↔ DIAL)")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "comm_usage.png"))
    plt.close()

    # --- Messages per episode ---
    plt.figure(figsize=(8, 5))
    plt.plot(hybrid_df["episode"], hybrid_df["msgs_per_ep"], label="Messages per Episode", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Total Messages Sent")
    plt.title("Communication Load")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "msgs_per_ep.png"))
    plt.close()

    # --- Message entropy (average per episode) ---
    plt.figure(figsize=(8, 5))
    plt.plot(hybrid_df["episode"], hybrid_df["msg_entropy_ep"], label="Message Entropy", color="red")
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
    plt.plot(hybrid_df["episode"], hybrid_df["avg_pair_dist"], label="Avg Pairwise Distance", color="purple")
    plt.xlabel("Episode")
    plt.ylabel("Distance")
    plt.title("Agent Spreading Behavior")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "avg_pair_dist.png"))
    plt.close()

    # --- First vs Last 100 episodes (summary bar plot) ---
    first100 = hybrid_df[hybrid_df["episode"] <= 100].mean(numeric_only=True)
    last100 = hybrid_df[hybrid_df["episode"] > hybrid_df["episode"].max() - 100].mean(numeric_only=True)

    metrics = ["return_mean", "msgs_per_ep", "pct_comm_on", "msg_entropy_ep", "avg_pair_dist"]
    plt.figure(figsize=(10, 6))
    x = range(len(metrics))
    plt.bar([i - 0.2 for i in x], [first100[m] for m in metrics], width=0.4, label="First 100")
    plt.bar([i + 0.2 for i in x], [last100[m] for m in metrics], width=0.4, label="Last 100")
    plt.xticks(x, metrics, rotation=30)
    plt.title("First vs Last 100 Episodes (Improvement & Switching Proof)")
    plt.legend()
    plt.grid(True, axis="y")
    plt.savefig(os.path.join(out_dir, "first_vs_last.png"))
    plt.close()

    # --- Console summary ---
    pct_ctde_only = (hybrid_df["pct_comm_on"] == 0.0).mean() * 100
    pct_dial_used = 100 - pct_ctde_only
    print("✅ Analysis complete. Plots saved in:", out_dir)
    print(f"CTDE-only episodes: {pct_ctde_only:.1f}%")
    print(f"DIAL used episodes: {pct_dial_used:.1f}%")

    # === Baseline Comparisons (if CSVs provided) ===
    dfs = [hybrid_df]
    if dial_csv: dfs.append(load_results(dial_csv, "DIAL-only", "red"))
    if ctde_csv: dfs.append(load_results(ctde_csv, "CTDE-only", "blue"))
    if random_csv: dfs.append(load_results(random_csv, "Random Switching", "purple"))

    if len(dfs) > 1:
        all_data = pd.concat(dfs, ignore_index=True)

        # Figure 1: Learning Curve Comparison
        plt.figure(figsize=(8,5))
        for label, group in all_data.groupby("label"):
            plt.plot(group["episode"], group["return_mean"].rolling(50).mean(),
                     label=label, color=group["color"].iloc[0])
        plt.xlabel("Episodes")
        plt.ylabel("Mean Return")
        plt.title("Learning Curves (Baseline Comparison)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "learning_curve_baselines.png"))
        plt.close()

        # Figure 2: Communication Overhead Comparison
        plt.figure(figsize=(8,5))
        for label, group in all_data.groupby("label"):
            overhead = group["msgs_per_ep"] / group["steps"]
            plt.plot(group["episode"], overhead.rolling(50).mean(),
                     label=label, color=group["color"].iloc[0])
        plt.xlabel("Episodes")
        plt.ylabel("Messages per Agent per Step")
        plt.title("Communication Overhead (Baseline Comparison)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "comm_overhead_baselines.png"))
        plt.close()

        print("✅ Baseline comparison plots saved in:", out_dir)


if __name__ == "__main__":
    # Only Hybrid by default
    analyze_results(
        hybrid_csv="results_hybrid/metrics.csv",
        dial_csv=None,      # e.g. "results_dial/metrics.csv"
        ctde_csv=None,      # e.g. "results_ctde/metrics.csv"
        random_csv=None     # e.g. "results_random/metrics.csv"
    )
