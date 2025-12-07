import os
import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_rewards(csv_path: str, out_path: str) -> None:
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if "episode" not in df.columns or "reward" not in df.columns:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], df["reward"], alpha=0.3, label="Episode Reward")
    if "average_reward_50" in df.columns:
        plt.plot(df["episode"], df["average_reward_50"], linewidth=2, label="Running Avg (50)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    _ensure_dir(out_path)
    plt.savefig(out_path)
    plt.close()


def plot_losses(csv_path: str, out_path: str) -> None:
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if "step" not in df.columns or "loss" not in df.columns:
        return
    plt.figure()
    plt.plot(df["step"], df["loss"], label="Loss")
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    _ensure_dir(out_path)
    plt.savefig(out_path)
    plt.close()


def plot_epsilon(csv_path: str, out_path: str) -> None:
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if "episode" not in df.columns or "epsilon" not in df.columns:
        return
    plt.figure()
    plt.plot(df["episode"], df["epsilon"], label="Epsilon")
    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True)
    plt.legend()
    _ensure_dir(out_path)
    plt.savefig(out_path)
    plt.close()


def plot_winrate(csv_path: str, out_path: str) -> None:
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    # support new header "winrate"
    col_name = "winrate" if "winrate" in df.columns else "win_rate"
    required = {"episode", col_name}
    if not required.issubset(set(df.columns)):
        return
    plt.figure()
    plt.plot(df["episode"], df[col_name], label="DDQN Win Rate")
    plt.title("Win Rate vs Random")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.grid(True)
    plt.legend()
    _ensure_dir(out_path)
    plt.savefig(out_path)
    plt.close()
