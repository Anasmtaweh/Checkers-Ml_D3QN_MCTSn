import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_loss_smooth(path, out_path, window=1000):
    df = pd.read_csv(path)
    df["loss_smooth"] = df["loss"].rolling(window=window).mean()

    plt.figure(figsize=(12, 5))
    plt.plot(df["step"], df["loss"], alpha=0.1, label="Raw Loss")
    plt.plot(df["step"], df["loss_smooth"], color="orange", label=f"Smoothed ({window})")
    plt.title("Training Loss (Smoothed)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()
