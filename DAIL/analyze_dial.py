# analyze_dial.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.plot_utils import (
    plot_learning_curve,
    plot_grouped_bars_first_last,
)

RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "dial_simple_spread_yaml.csv")
SUMMARY_OUT = os.path.join(RESULTS_DIR, "dial_summary_yaml.csv")
FIG_LEARNING = os.path.join(RESULTS_DIR, "learning_curves_yaml.png")
FIG_BARS = os.path.join(RESULTS_DIR, "first_last_bars_yaml.png")

# -------- which metrics to analyze --------
CURVE_METRICS = [
    ("return_mean", "Mean episode reward (↑ better)"),
    ("return_sum", "Total team reward (↑ better)"),
    ("msgs_per_ep", "Messages per episode"),
]
BAR_METRICS = [
    ("return_mean", "Mean reward"),
    ("return_sum", "Team reward"),
    ("reward_a0", "Agent0 reward"),
    ("reward_a1", "Agent1 reward"),
    ("reward_a2", "Agent2 reward"),
]

def mean_first_last(df: pd.DataFrame, col: str, first_n=100, last_n=100):
    if df.empty or col not in df:
        return np.nan, np.nan, np.nan
    df = df.sort_values("episode")
    first = df.head(first_n)[col].mean()
    last = df.tail(last_n)[col].mean()
    delta = last - first
    return first, last, delta

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}. Run training first.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # -------- Learning curves (line plots) --------
    fig, axes = plt.subplots(len(CURVE_METRICS), 1, figsize=(8, 4 * len(CURVE_METRICS)))
    if len(CURVE_METRICS) == 1:
        axes = [axes]
    for ax, (col, label) in zip(axes, CURVE_METRICS):
        if col in df:
            plot_learning_curve(df["episode"].values, df[col].values, label, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG_LEARNING, dpi=150)
    plt.close(fig)

    # -------- First 100 vs Last 100 summary --------
    rows = []
    for col, pretty in BAR_METRICS:
        f, l, d = mean_first_last(df, col, 100, 100)
        rows.append({
            "metric": col,
            "label": pretty,
            "mean_first_100": f,
            "mean_last_100": l,
            "delta_last_minus_first": d
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_OUT, index=False)

    # -------- Bar chart (first vs last) --------
    labels = [r["label"] for r in rows]
    first_vals = [r["mean_first_100"] for r in rows]
    last_vals = [r["mean_last_100"] for r in rows]
    fig2 = plot_grouped_bars_first_last(labels, first_vals, last_vals, title="First 100 vs Last 100 (means)")
    fig2.savefig(FIG_BARS, dpi=150)
    plt.close(fig2)

    # -------- Console report --------
    print("\n=== DIAL Simple Spread — First vs Last 100 Episodes ===")
    for r in rows:
        print(f"{r['label']:<18}  first={r['mean_first_100']:.3f}  last={r['mean_last_100']:.3f}  Δ={r['delta_last_minus_first']:.3f}")
    print(f"\nSaved:\n  - Learning curves: {FIG_LEARNING}\n  - First/Last bars: {FIG_BARS}\n  - Summary CSV:     {SUMMARY_OUT}")

if __name__ == "__main__":
    main()
