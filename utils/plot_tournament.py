import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_tournament_results(json_path, output_path):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    results = data.get("results", {})
    
    # Sort by overall win rate descending (Best at top for horizontal plot)
    # We actually want Best at *Bottom* for barh if we plot 0..N, so let's sort ascending
    # But standard list usage: 0 is bottom. So let's sort Ascending by Overall WR.
    sorted_agents = sorted(results.items(), key=lambda x: x[1]['overall_wr'], reverse=False)
    
    agents = [k for k, v in sorted_agents]
    p1_wrs = [v['p1_wr'] for k, v in sorted_agents]
    p2_wrs = [v['p2_wr'] for k, v in sorted_agents]
    overall_wrs = [v['overall_wr'] for k, v in sorted_agents]

    # Plot Settings
    plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
    fig, ax = plt.subplots(figsize=(14, 10))
    
    y = np.arange(len(agents))
    height = 0.35  # Bar height

    # Colors: Professional Red (P1) and Slate Black (P2)
    color_p1 = '#E57373'  # Muted Red
    color_p2 = '#455A64'  # Blue-Grey/Black
    color_overall = '#FFB74D' # Muted Orange

    # Horizontal Bars
    rects1 = ax.barh(y + height/2, p1_wrs, height, label='Player 1 (Red)', color=color_p1, alpha=0.9, edgecolor='white')
    rects2 = ax.barh(y - height/2, p2_wrs, height, label='Player 2 (Black)', color=color_p2, alpha=0.9, edgecolor='white')

    # Add Overall Win Rate Dots
    # ax.plot(overall_wrs, y, 'o', color=color_overall, label='Overall WR', markersize=8, markeredgecolor='white')

    # Styling
    ax.set_xlabel('Win Rate (%)', fontsize=12, fontweight='bold', color='#333')
    ax.set_title('Tournament Results: Win Rates by Side', fontsize=16, fontweight='bold', pad=20, color='#333')
    ax.set_yticks(y)
    ax.set_yticklabels(agents, fontsize=11)
    ax.set_xlim(0, 105)  # Give room for labels
    ax.legend(loc='lower right', frameon=True, shadow=True)

    # Grid
    ax.grid(axis='x', linestyle='--', alpha=0.4, color='#999')
    ax.set_axisbelow(True)  # Grid behind bars

    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#ccc')
    ax.spines['bottom'].set_color('#ccc')

    # Value Labels
    def add_labels(rects, is_p1):
        for rect in rects:
            width = rect.get_width()
            y_pos = rect.get_y() + rect.get_height() / 2
            
            # Smart label positioning
            label_x = width + 1
            color = 'black'
            
            # Bold the text if it's a high win rate
            weight = 'bold' if width > 50 else 'normal'
            
            ax.text(label_x, y_pos, f'{width:.1f}%', 
                    va='center', fontsize=9, color=color, fontweight=weight)

    add_labels(rects1, True)
    add_labels(rects2, False)

    # Highlight the Best Agent (Last one in list)
    ax.get_yticklabels()[-1].set_fontweight('bold')
    ax.get_yticklabels()[-1].set_color('#2E7D32')  # Green text for winner

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Enhanced plot saved to {output_path}")

if __name__ == "__main__":
    # Determine paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir) 
    
    if os.path.exists("fair_tournament_results.json"):
        json_file = "fair_tournament_results.json"
        out_file = "tournament_plot.png"
    else:
        json_file = os.path.join(project_root, "fair_tournament_results.json")
        out_file = os.path.join(project_root, "tournament_plot.png")

    plot_tournament_results(json_file, out_file)