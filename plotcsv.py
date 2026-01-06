import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_simple_csv(file_path, plot_title="Simple Training Log"):
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # Read the CSV
    # skipinitialspace=True fixes errors if there are spaces after commas
    df = pd.read_csv(file_path, skipinitialspace=True)

    # Clean up column names (remove extra spaces)
    df.columns = df.columns.str.strip()

    # --- SMOOTHING (Moving Average) ---
    # We smooth the lines slightly so they aren't too jagged.
    # We use a window of 20. You can change this number.
    window = 20
    df['Reward_Smooth'] = df['Reward'].rolling(window=window).mean()
    df['Loss_Smooth']   = df['Loss'].rolling(window=window).mean()
    
    # Setup the plot: 2 rows, 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(plot_title, fontsize=16)

    # 1. WIN RATES (Top Left)
    # This shows your P1, P2, and VsRandom win percentages
    axs[0, 0].plot(df['Episode'], df['P1_WinRate'], label='P1 Win %', color='red')
    axs[0, 0].plot(df['Episode'], df['vsRandom_WinRate'], label='vs Random %', color='blue')
    axs[0, 0].set_title("Win Rates")
    axs[0, 0].set_ylabel("Win %")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. REWARD (Top Right)
    # The actual points the agent gets
    axs[0, 1].plot(df['Episode'], df['Reward_Smooth'], color='green', label='Avg Reward')
    axs[0, 1].set_title("Average Reward")
    axs[0, 1].set_ylabel("Points")
    axs[0, 1].grid(True)

    # 3. LOSS (Bottom Left)
    # The error rate of the neural network
    axs[1, 0].plot(df['Episode'], df['Loss_Smooth'], color='orange', label='Loss')
    axs[1, 0].set_title("Training Loss")
    axs[1, 0].set_ylabel("Error")
    axs[1, 0].grid(True)

    # 4. Q-VALUES (Bottom Right)
    # Shows if the AI thinks it can get high scores
    if 'MaxQ' in df.columns:
        axs[1, 1].plot(df['Episode'], df['MaxQ'], color='purple', label='Max Q (Best Move Score)')
        axs[1, 1].set_title("Q-Values (Confidence)")
        axs[1, 1].set_ylabel("Estimated Value")
        axs[1, 1].legend()
        axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

# ==========================================
# RUN IT HERE
# ==========================================

# Just change the file path below
plot_simple_csv(
    file_path="C:/Users/YourName/logs.csv", 
    plot_title="My D3QN Training"
)