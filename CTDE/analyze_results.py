# analyze_results.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze(csv_path):
    df = pd.read_csv(csv_path)

    # Plot learning curve
    plt.figure(figsize=(8, 5))
    plt.plot(df["episode"], df["return_mean"], label="Return Mean")
    plt.xlabel("Episode")
    plt.ylabel("Mean Return")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_curve.png")
    plt.close()

    # First 100 vs last 100
    first_mean = df["return_mean"][:100].mean()
    last_mean = df["return_mean"][-100:].mean()

    plt.figure(figsize=(5, 5))
    plt.bar(["First 100", "Last 100"], [first_mean, last_mean], color=["red", "green"])
    plt.ylabel("Mean Return")
    plt.title("Performance Improvement")
    plt.savefig("first_last_bar.png")
    plt.close()

    print(f"First 100 mean return: {first_mean:.3f}")
    print(f"Last 100 mean return: {last_mean:.3f}")
    print(f"Improvement: {last_mean - first_mean:.3f}")

if __name__ == "__main__":
    analyze("results_ctde/ctde_simple_spread.csv")
