import pandas as pd
import matplotlib.pyplot as plt
import io

# 1. YOUR RAW DATA
csv_data = """iteration,az_wins,d3qn_wins,draws,win_rate,draw_rate,loss_rate,p1_wins,p2_wins
200,10,0,10,0.5,0.5,0.0,0,10
201,20,0,0,1.0,0.0,0.0,10,10
202,0,0,20,0.0,1.0,0.0,0,0
203,0,0,20,0.0,1.0,0.0,0,0
204,10,0,10,0.5,0.5,0.0,0,10
205,10,0,10,0.5,0.5,0.0,0,10
206,20,0,0,1.0,0.0,0.0,10,10
207,20,0,0,1.0,0.0,0.0,10,10
208,20,0,0,1.0,0.0,0.0,10,10
209,20,0,0,1.0,0.0,0.0,10,10
210,10,0,10,0.5,0.5,0.0,0,10
211,0,0,20,0.0,1.0,0.0,0,0
212,20,0,0,1.0,0.0,0.0,10,10
213,20,0,0,1.0,0.0,0.0,10,10
214,20,0,0,1.0,0.0,0.0,10,10
215,10,0,10,0.5,0.5,0.0,10,0
216,10,0,10,0.5,0.5,0.0,0,10
217,20,0,0,1.0,0.0,0.0,10,10
218,0,0,20,0.0,1.0,0.0,0,0
219,10,0,10,0.5,0.5,0.0,10,0
220,10,0,10,0.5,0.5,0.0,0,10
221,20,0,0,1.0,0.0,0.0,10,10
222,20,0,0,1.0,0.0,0.0,10,10
223,20,0,0,1.0,0.0,0.0,10,10
224,10,0,10,0.5,0.5,0.0,0,10
225,20,0,0,1.0,0.0,0.0,10,10
226,20,0,0,1.0,0.0,0.0,10,10
227,0,0,20,0.0,1.0,0.0,0,0
228,0,0,20,0.0,1.0,0.0,0,0
229,20,0,0,1.0,0.0,0.0,10,10
"""

# 2. LOAD DATA
df = pd.read_csv(io.StringIO(csv_data))

# 3. SMOOTHING (Essential for this data)
# Window 3 averages the last 3 points to stop the lines from looking like a barcode
window = 3
df['win_smooth'] = df['win_rate'].rolling(window=window, min_periods=1).mean()
df['draw_smooth'] = df['draw_rate'].rolling(window=window, min_periods=1).mean()
df['loss_smooth'] = df['loss_rate'].rolling(window=window, min_periods=1).mean()

# 4. PLOTTING
plt.figure(figsize=(12, 6), dpi=150)

# Colors
GREEN = '#2ca02c'
RED = '#d62728'
GRAY = '#7f7f7f'

# --- Plot Wins ---
plt.plot(df['iteration'], df['win_smooth'], color=GREEN, linewidth=2.5, label='AlphaZero Win Rate')
plt.fill_between(df['iteration'], 0, df['win_smooth'], color=GREEN, alpha=0.1) # Green Glow
plt.scatter(df['iteration'], df['win_rate'], color=GREEN, alpha=0.2, s=20) # Raw dots

# --- Plot Draws ---
plt.plot(df['iteration'], df['draw_smooth'], color=GRAY, linewidth=2, linestyle='--', label='Draw Rate')

# --- Plot Losses ---
plt.plot(df['iteration'], df['loss_smooth'], color=RED, linewidth=2, linestyle=':', label='D3QN Win Rate')
plt.scatter(df['iteration'], df['loss_rate'], color=RED, alpha=0.2, s=20) # Raw dots

# Formatting
plt.title("AlphaZero Performance: Iterations 200-229", fontsize=14, fontweight='bold')
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Rate", fontsize=12)
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

# Y-Axis Percentages
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])

# Save
output_path = "plot_200_229.png"
plt.savefig(output_path)
print(f"Plot saved to {output_path}")