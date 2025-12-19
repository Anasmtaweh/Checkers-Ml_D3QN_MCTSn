import matplotlib.pyplot as plt
import re
import numpy as np

def plot_logs(log_file="training_log.txt"):
    episodes = []
    rewards = []
    losses = []
    lengths = []
    outcomes = []  # 1 for Win, 0 for Loss/Draw
    
    print(f"📊 Reading {log_file}...")
    
    try:
        with open(log_file, "r") as f:
            for line in f:
                if "Episode" in line and "Reward" in line:
                    try:
                        # Extract Data using Regex
                        ep_match = re.search(r"Episode\s+(\d+)", line)
                        rew_match = re.search(r"Reward:\s+([-\d.eE+]+)", line)
                        len_match = re.search(r"Length:\s+([-\d.eE+]+)", line)
                        loss_match = re.search(r"Loss:\s+([-\d.eE+]+)", line)
                        
                        if ep_match and rew_match and loss_match and len_match:
                            ep = int(ep_match.group(1))
                            rew = float(rew_match.group(1))
                            length = float(len_match.group(1))
                            loss = float(loss_match.group(1))
                            
                            # Extract Outcome (Simple string check)
                            # We treat WIN as 1, everything else as 0 for the win rate calc
                            is_win = 1 if "WIN" in line else 0
                            
                            episodes.append(ep)
                            rewards.append(rew)
                            lengths.append(length)
                            losses.append(loss)
                            outcomes.append(is_win)
                    except:
                        continue
    except FileNotFoundError:
        print(f"❌ Error: Log file '{log_file}' not found.")
        print("   Please run main.py first to generate training logs.")
        return

    if not episodes:
        print("❌ No data found! The log file might be empty.")
        return

    # --- SMOOTHING (Moving Average) ---
    # We use a window to smooth out the noise and see the trend
    window = 50
    if len(episodes) < window:
        window = 1  # Not enough data to smooth yet

    def moving_avg(data):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    avg_rewards = moving_avg(rewards)
    avg_lengths = moving_avg(lengths)
    avg_losses = moving_avg(losses)
    avg_win_rate = moving_avg(outcomes) * 100.0  # Convert to Percentage

    # Adjust x-axis to match the smoothed data length
    ma_episodes = episodes[len(episodes)-len(avg_rewards):]

    # --- PLOTTING ---
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'D3QN Training Dashboard (Smoothed window={window})', fontsize=16)
    
    # 1. Win Rate (The most important metric)
    axs[0, 0].plot(ma_episodes, avg_win_rate, color='green', linewidth=2)
    axs[0, 0].set_title('Win Rate (%)')
    axs[0, 0].set_ylabel('Win %')
    axs[0, 0].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label="50% Break-even")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()
    
    # 2. Average Reward
    axs[0, 1].plot(ma_episodes, avg_rewards, color='blue', linewidth=1.5)
    axs[0, 1].set_title('Average Reward')
    axs[0, 1].set_ylabel('Reward')
    axs[0, 1].grid(True, alpha=0.3)
    
    # 3. Training Loss
    axs[1, 0].plot(ma_episodes, avg_losses, color='red', linewidth=1)
    axs[1, 0].set_title('Training Loss (Log Scale)')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_yscale('log')
    axs[1, 0].grid(True, alpha=0.3)
    
    # 4. Game Length
    axs[1, 1].plot(ma_episodes, avg_lengths, color='purple', linewidth=1.5)
    axs[1, 1].set_title('Average Game Length')
    axs[1, 1].set_ylabel('Steps')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_dashboard.png", dpi=300) # High resolution for report
    print("✅ Graph saved as 'training_dashboard.png'")
    plt.show()

if __name__ == "__main__":
    plot_logs()
