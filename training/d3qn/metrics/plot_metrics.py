import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# Utility: Moving Average Smoothing
# -------------------------------------------------------------------------
def smooth(data, window=100):
    """Return a moving average of a 1D list/array."""
    return pd.Series(data).rolling(window=window, min_periods=1).mean()


# -------------------------------------------------------------------------
# Loss Plot
# -------------------------------------------------------------------------
def plot_loss(loss_csv, out_path, window=1000):
    df = pd.read_csv(loss_csv)

    plt.figure(figsize=(12, 5))
    plt.plot(df["step"], df["loss"], alpha=0.15, label="Raw Loss")
    plt.plot(df["step"], smooth(df["loss"], window), label=f"Smoothed Loss ({window})", color="orange")

    plt.title("Training Loss (Smoothed)", fontsize=14)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot saved] {out_path}")


# -------------------------------------------------------------------------
# Reward Plot
# -------------------------------------------------------------------------
def plot_rewards(reward_csv, out_path, window=200):
    df = pd.read_csv(reward_csv)

    plt.figure(figsize=(14, 6))

    # raw rewards (transparent blue)
    plt.plot(df["episode"], df["reward"], alpha=0.2, label="Episode Reward", color="skyblue")

    # smoothed rewards (thicker orange)
    plt.plot(df["episode"], smooth(df["reward"], window), label=f"Smoothed Reward ({window})", color="orange")

    plt.title("Episode Rewards (Smoothed)", fontsize=14)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot saved] {out_path}")


# -------------------------------------------------------------------------
# Epsilon Decay Plot
# -------------------------------------------------------------------------
def plot_epsilon(episode_csv, out_path):
    df = pd.read_csv(episode_csv)

    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], df["epsilon"], label="Epsilon", color="blue")

    plt.title("Epsilon Decay", fontsize=14)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot saved] {out_path}")


# -------------------------------------------------------------------------
# Winrate Plot (Smoothed + Fixed Scaling)
# -------------------------------------------------------------------------
def plot_winrate(winrate_csv, out_path, window=5):
    df = pd.read_csv(winrate_csv)

    # raw win rate
    plt.figure(figsize=(12, 5))
    plt.plot(df["episode"], df["win_rate"], alpha=0.3, label="Raw Winrate")

    # smoothed winrate
    plt.plot(df["episode"], smooth(df["win_rate"], window), label=f"Smoothed Winrate ({window})", color="orange")

    plt.ylim(0.85, 1.01)   # avoid tall spikes and jitter
    plt.title("Win Rate vs Random (Smoothed)", fontsize=14)
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot saved] {out_path}")


# -------------------------------------------------------------------------
# Combined Function (Optional convenience)
# -------------------------------------------------------------------------
def plot_all_metrics(base_dir="logs/d3qn"):
    """
    Automatically generates all plots if expected CSVs exist.
    """

    episode_csv = f"{base_dir}/episode_stats.csv"
    loss_csv = f"{base_dir}/loss.csv"
    winrate_csv = f"{base_dir}/metrics/winrate.csv"

    plot_rewards(episode_csv, f"{base_dir}/plots/reward_curve.png")
    plot_epsilon(episode_csv, f"{base_dir}/plots/epsilon_curve.png")
    plot_loss(loss_csv, f"{base_dir}/plots/loss_curve.png")
    plot_winrate(winrate_csv, f"{base_dir}/plots/winrate_curve.png")

    print("\nAll metric plots generated successfully.\n")
