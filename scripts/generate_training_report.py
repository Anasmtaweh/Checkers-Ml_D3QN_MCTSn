#!/usr/bin/env python3
"""
generate_training_report.py - Generate Training Report

Creates a comprehensive training report with:
- Summary statistics
- All training plots
- Performance analysis
- Recommendations

Outputs: HTML report (can be converted to PDF)
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime


def generate_report(log_file: str, output_file: str = "training_report.html"):
    """Generate HTML training report."""
    
    # Load data
    df = pd.read_csv(log_file)
    
    # Calculate statistics
    total_iters = len(df)
    final_p1_wr = df['p1_win_rate'].iloc[-1] * 100
    final_p2_wr = df['p2_win_rate'].iloc[-1] * 100
    best_p1_wr = df['p1_win_rate'].max() * 100
    best_p1_idx = int(df['p1_win_rate'].idxmax())
    best_p1_iter = int(df.iloc[best_p1_idx]['iteration'])
    
    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('AlphaZero Training Report', fontsize=16, fontweight='bold')
    
    # Win rates
    axes[0, 0].plot(df['iteration'], df['p1_win_rate']*100, label='P1')
    axes[0, 0].plot(df['iteration'], df['p2_win_rate']*100, label='P2')
    axes[0, 0].set_title('Win Rates')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Losses
    axes[0, 1].plot(df['iteration'], df['total_loss'], label='Total')
    axes[0, 1].plot(df['iteration'], df['value_loss'], label='Value', alpha=0.7)
    axes[0, 1].plot(df['iteration'], df['policy_loss'], label='Policy', alpha=0.7)
    axes[0, 1].set_title('Training Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Game length
    axes[1, 0].plot(df['iteration'], df['avg_game_length'])
    axes[1, 0].set_title('Average Game Length')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Buffer size
    if 'buffer_size' in df.columns:
        axes[1, 1].plot(df['iteration'], df['buffer_size'])
        axes[1, 1].set_title('Buffer Size Growth')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_file.replace('.html', '_plots.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AlphaZero Training Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2E86AB; }}
            .stat {{ background: #f0f0f0; padding: 10px; margin: 10px 0; }}
            img {{ max-width: 100%; }}
        </style>
    </head>
    <body>
        <h1>AlphaZero Training Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Summary</h2>
        <div class="stat">Total Iterations: {total_iters}</div>
        <div class="stat">Final P1 Win Rate: {final_p1_wr:.1f}%</div>
        <div class="stat">Final P2 Win Rate: {final_p2_wr:.1f}%</div>
        <div class="stat">Best P1 Win Rate: {best_p1_wr:.1f}% (Iteration {best_p1_iter})</div>
        
        <h2>Training Plots</h2>
        <img src="{os.path.basename(plot_file)}" alt="Training Plots">
        
        <h2>Recommendations</h2>
        <ul>
            <li>{'✓ Training converged well' if final_p1_wr > 60 else '⚠ Consider more iterations'}</li>
            <li>{'✓ Balanced win rates' if abs(final_p1_wr - final_p2_wr) < 10 else '⚠ Imbalanced players'}</li>
        </ul>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"✓ Report generated: {output_file}")
    print(f"✓ Plots saved: {plot_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate training report')
    parser.add_argument('--log-file', default='data/training_logs/alphazero_training.csv')
    parser.add_argument('--output', default='training_report.html')
    args = parser.parse_args()
    
    generate_report(args.log_file, args.output)
