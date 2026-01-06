import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import seaborn as sns

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILE = "/home/anas/ML_Gen2/data/training_logs/alphazero_training.csv"
OUTPUT_IMG = "training_technical_metrics.png"

# Same Events for Context
EVENTS = {
    133: "600 Sims",
    142: "Wide Vision",
    155: "Buffer Surgery",
    167: "Peak Init.",
    176: "800 Sims",
    182: "Bias -0.3",
    186: "Hybrid",
    190: "Aggro",
    200: "End"
}

def plot_tech():
    if not os.path.exists(CSV_FILE):
        print(f"❌ File not found: {CSV_FILE}")
        return

    # Scientific Style
    sns.set_theme(style="whitegrid", rc={"grid.linestyle": ":"})
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Load Data
    df = pd.read_csv(CSV_FILE)
    
    # Create Canvas (2 Plots: Loss & Time)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
    plt.subplots_adjust(hspace=0.15)
    
    # ---------------------------------------------------------
    # PLOT 1: LOSS CONVERGENCE (Stability)
    # ---------------------------------------------------------
    # Total Loss
    ax1.plot(df['iteration'], df['total_loss'], color='forestgreen', linewidth=2, label='Total Loss')
    
    # Policy Loss (The Strategy)
    ax1.plot(df['iteration'], df['policy_loss'], color='orange', linestyle='--', linewidth=1.5, label='Policy Loss')
    
    # Value Loss (The Score Prediction) - usually very small, so we might skip or scale it
    # ax1.plot(df['iteration'], df['value_loss'], color='blue', linestyle=':', label='Value Loss')

    ax1.set_ylabel("Loss Value", fontsize=12, fontweight='bold')
    ax1.set_title("Network Convergence (Lower is More Stable)", fontsize=16, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', frameon=True, shadow=True)
    
    # Zoom in to ignore the huge spike at Iter 1
    if len(df) > 5:
        ax1.set_ylim(0.4, 1.2) # Focused view of the "steady state"

    # ---------------------------------------------------------
    # PLOT 2: COMPUTATIONAL COST (Time)
    # ---------------------------------------------------------
    # Convert seconds to Minutes for readability
    time_min = df['elapsed_time_s'] / 60.0
    
    ax2.plot(df['iteration'], time_min, color='darkslategray', linewidth=2, label='Mins per Iteration')
    
    # Fill area to look cool
    ax2.fill_between(df['iteration'], time_min, color='gray', alpha=0.1)

    ax2.set_ylabel("Minutes", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Training Iteration", fontsize=12, fontweight='bold')
    ax2.set_title("Computational Cost (Hardware Load)", fontsize=14, fontweight='bold', pad=10)
    ax2.legend(loc='upper left')

    # ---------------------------------------------------------
    # AXIS & ANNOTATIONS
    # ---------------------------------------------------------
    major_locator = ticker.MultipleLocator(10)
    minor_locator = ticker.MultipleLocator(2)
    ax2.xaxis.set_major_locator(major_locator)
    ax2.xaxis.set_minor_locator(minor_locator)
    ax2.set_xlim(0, df['iteration'].max() + 5)

    # DRAW EVENTS (Staggered Labels to fix overlap)
    for i, (iter_num, label) in enumerate(EVENTS.items()):
        if iter_num <= df['iteration'].max():
            for ax in [ax1, ax2]:
                ax.axvline(x=iter_num, color='red', linestyle=':', linewidth=1, alpha=0.5)
            
            # Stagger text heights: Low, High, Low, High...
            # This prevents them from writing over each other
            y_pos = ax2.get_ylim()[1] * (0.85 if i % 2 == 0 else 0.95)
            
            ax2.text(iter_num, y_pos, f"{label}", 
                     rotation=90, fontsize=9, fontweight='bold', color='maroon', 
                     ha='right', va='top')

    # Save
    plt.savefig(OUTPUT_IMG, dpi=200, bbox_inches='tight')
    print(f"✅ Technical Dashboard saved to: {OUTPUT_IMG}")

if __name__ == "__main__":
    try:
        plot_tech()
    except ImportError:
        print("❌ Error: Libraries missing.")