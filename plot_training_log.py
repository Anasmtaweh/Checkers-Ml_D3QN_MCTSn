import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import seaborn as sns

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILE = "/home/anas/ML_Gen2/data/training_logs/alphazero_training.csv"
OUTPUT_IMG = "training_dashboard_pro.png"

# Mapped strictly to your Technical Documentation
EVENTS = {
    133: "Era 2: Depth Pivot (600 Sims)",
    142: "Era 4: Wide Vision (High Noise)",
    155: "⚠️ EVENT: Buffer Surgery",
    167: "Era 6: Initiative Peak",
    176: "Era 7: GM Ascent (800 Sims)",
    182: "Era 7.5: Bias Increase (-0.3)",
    186: "Era 8: Chaos Hybrid",
    190: "Policy: Aggressive Opt.",
    194: "Policy: Reactive Opt.",
    200: "Final Convergence"
}

def plot_pro():
    if not os.path.exists(CSV_FILE):
        print(f"❌ File not found: {CSV_FILE}")
        return

    # Set Style (Scientific/Clean)
    sns.set_theme(style="whitegrid", rc={"grid.linestyle": ":"})
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Load Data
    df = pd.read_csv(CSV_FILE)
    
    # Create Canvas
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14), sharex=True)
    plt.subplots_adjust(hspace=0.1)
    
    # ---------------------------------------------------------
    # PLOT 1: POLICY DISTRIBUTION (Win Rates)
    # ---------------------------------------------------------
    # Raw Data (Faint background)
    ax1.plot(df['iteration'], df['p1_win_rate'], color='crimson', alpha=0.1, linewidth=1)
    ax1.plot(df['iteration'], df['p2_win_rate'], color='black', alpha=0.1, linewidth=1)
    
    # Smoothed Trends (Rolling Window = 5)
    window = 5
    ax1.plot(df['iteration'], df['p1_win_rate'].rolling(window, min_periods=1).mean(), 
             color='crimson', label='P1 Win Rate (Initiative)', linewidth=2.5)
    ax1.plot(df['iteration'], df['p2_win_rate'].rolling(window, min_periods=1).mean(), 
             color='black', label='P2 Win Rate (Reaction)', linewidth=2.5)
    ax1.plot(df['iteration'], df['draw_rate'].rolling(window, min_periods=1).mean(), 
             color='steelblue', label='Draw Rate (Equilibrium)', linestyle='--', linewidth=2, alpha=0.9)

    ax1.set_ylabel("Probability", fontsize=12, fontweight='bold')
    ax1.set_title("Policy Convergence & Metagame Shifts", fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='center right', frameon=True, shadow=True, fontsize=11)
    ax1.set_ylim(-0.05, 1.05)

    # ---------------------------------------------------------
    # PLOT 2: SEARCH COMPLEXITY (Game Length)
    # ---------------------------------------------------------
    ax2.plot(df['iteration'], df['avg_game_length'], color='rebeccapurple', alpha=0.15) # Raw
    ax2.plot(df['iteration'], df['avg_game_length'].rolling(window, min_periods=1).mean(), 
             color='rebeccapurple', linewidth=2.5, label='Avg Moves per Game')
    
    ax2.set_ylabel("Moves", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Training Iteration", fontsize=12, fontweight='bold')
    ax2.set_title("Game Complexity (Depth Metric)", fontsize=14, fontweight='bold', pad=10)
    ax2.legend(loc='upper left', frameon=True, fontsize=11)

    # ---------------------------------------------------------
    # AXIS FORMATTING
    # ---------------------------------------------------------
    # Major ticks every 10 iterations for precision reading
    major_locator = ticker.MultipleLocator(10)
    minor_locator = ticker.MultipleLocator(2)
    
    ax2.xaxis.set_major_locator(major_locator)
    ax2.xaxis.set_minor_locator(minor_locator)
    ax2.set_xlim(0, df['iteration'].max() + 5)
    
    # ---------------------------------------------------------
    # TECHNICAL ANNOTATIONS
    # ---------------------------------------------------------
    # We alternate text height to prevent overlapping
    text_heights = [1.02, 1.07] 
    
    for i, (iter_num, label) in enumerate(EVENTS.items()):
        if iter_num <= df['iteration'].max():
            # Draw vertical line across both plots
            for ax in [ax1, ax2]:
                ax.axvline(x=iter_num, color='gray', linestyle='-', linewidth=1, alpha=0.5)
            
            # Label on top plot only
            h = text_heights[i % 2] # Toggle height
            ax1.text(iter_num, h, f" {iter_num}: {label}", 
                     rotation=45, fontsize=10, fontweight='bold', color='#444', 
                     ha='left', va='bottom', transform=ax1.get_xaxis_transform())

    # Save
    plt.savefig(OUTPUT_IMG, dpi=200, bbox_inches='tight')
    print(f"✅ Professional Dashboard saved to: {OUTPUT_IMG}")

if __name__ == "__main__":
    try:
        plot_pro()
    except ImportError:
        print("❌ Error: Libraries missing. Run: pip install seaborn pandas matplotlib")