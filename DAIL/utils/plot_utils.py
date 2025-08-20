# utils/plot_utils.py
import matplotlib.pyplot as plt
import numpy as np

def moving_average(x, w=20):
    if len(x) == 0:
        return np.array([])
    w = max(1, int(w))
    cumsum = np.cumsum(np.insert(x, 0, 0))
    ma = (cumsum[w:] - cumsum[:-w]) / float(w)
    # pad to original length
    pad = np.full(w - 1, np.nan)
    return np.concatenate([pad, ma])

def plot_learning_curve(episodes, values, ylabel, ax=None, window=20):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, values, label="per-episode")
    ma = moving_average(values, w=window)
    ax.plot(episodes, ma, label=f"moving avg ({window})", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return ax

def plot_grouped_bars_first_last(labels, first_vals, last_vals, title="First vs Last"):
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, first_vals, width, label="First 100")
    ax.bar(x + width/2, last_vals, width, label="Last 100")

    ax.set_ylabel("Mean value")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
