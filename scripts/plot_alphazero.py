#!/usr/bin/env python3
"""
plot_alphazero.py - AlphaZero Training Visualization Dashboard

Real-time monitoring dashboard for AlphaZero training progress.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
import argparse
import os
import sys
import warnings
from datetime import datetime

# Suppress font warnings for missing emoji glyphs
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

DEFAULT_LOG_FILE = "data/training_logs/alphazero_training.csv"
REFRESH_INTERVAL = 30000  # milliseconds (30 seconds)

# Plot styling with better colors
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'p1': '#1f77b4',      # Blue
    'p2': '#d62728',      # Red
    'draws': '#ff7f0e',   # Orange
    'total_loss': '#e74c3c',     # Bright Red
    'value_loss': '#27ae60',     # Green
    'policy_loss': '#9b59b6',    # Purple
    'game_length': '#34495e',    # Dark Gray
    'buffer': '#95a5a6',         # Light Gray
}


# ════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════

def load_training_data(filepath: str) -> pd.DataFrame:
    """Load training data from CSV log."""
    if not os.path.exists(filepath):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(filepath)
        required_cols = [
            'iteration', 'p1_win_rate', 'p2_win_rate', 'draw_rate',
            'total_loss', 'value_loss', 'policy_loss', 'avg_game_length'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return pd.DataFrame()
        
        return df
    except Exception:
        return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ════════════════════════════════════════════════════════════════════

def plot_win_rates(ax, df: pd.DataFrame):
    """Plot win rates over iterations."""
    ax.clear()
    
    if df.empty or len(df) < 2:
        ax.text(0.5, 0.5, 'Waiting for more training data...\n(Need at least 2 iterations)', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=13, color='#7f8c8d', style='italic')
        ax.set_title('Win Rates Over Time', fontsize=15, fontweight='bold', pad=10)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.4, linestyle='--')
        return
    
    iterations = df['iteration']
    
    # Plot win rates with thicker lines
    ax.plot(iterations, df['p1_win_rate'] * 100, 
            label='Player 1', color=COLORS['p1'], linewidth=3, marker='o', markersize=6)
    ax.plot(iterations, df['p2_win_rate'] * 100, 
            label='Player 2', color=COLORS['p2'], linewidth=3, marker='s', markersize=6)
    ax.plot(iterations, df['draw_rate'] * 100, 
            label='Draws', color=COLORS['draws'], linewidth=3, marker='^', markersize=6)
    
    # Reference line at 50%
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.6, linewidth=1.5, label='50% Reference')
    
    # Styling
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Win Rates Over Time', fontsize=15, fontweight='bold', pad=10)
    ax.legend(loc='best', framealpha=0.95, fontsize=11, edgecolor='black')
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.set_ylim(0, 105)
    
    # Add current values in corner
    latest = df.iloc[-1]
    info_text = (
        f"Current (Iter {int(latest['iteration'])}):\n"
        f"P1: {latest['p1_win_rate']*100:.1f}% | "
        f"P2: {latest['p2_win_rate']*100:.1f}% | "
        f"Draw: {latest['draw_rate']*100:.1f}%"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='black'))


def plot_losses(ax, df: pd.DataFrame):
    """Plot training losses over iterations."""
    ax.clear()
    
    if df.empty or len(df) < 2:
        ax.text(0.5, 0.5, 'Waiting for more training data...\n(Need at least 2 iterations)', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=13, color='#7f8c8d', style='italic')
        ax.set_title('Training Losses', fontsize=15, fontweight='bold', pad=10)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.grid(True, alpha=0.4, linestyle='--')
        return
    
    iterations = df['iteration']
    
    # Plot losses
    ax.plot(iterations, df['total_loss'], 
            label='Total Loss', color=COLORS['total_loss'], linewidth=3, marker='o', markersize=6)
    ax.plot(iterations, df['value_loss'], 
            label='Value Loss', color=COLORS['value_loss'], linewidth=2.5, marker='s', markersize=5)
    ax.plot(iterations, df['policy_loss'], 
            label='Policy Loss', color=COLORS['policy_loss'], linewidth=2.5, marker='^', markersize=5)
    
    # Styling
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training Losses', fontsize=15, fontweight='bold', pad=10)
    ax.legend(loc='best', framealpha=0.95, fontsize=11, edgecolor='black')
    ax.grid(True, alpha=0.4, linestyle='--')
    
    # Use log scale if needed
    loss_range = df['total_loss'].max() - df['total_loss'].min()
    if loss_range > 10:
        ax.set_yscale('log')
        ax.set_ylabel('Loss (log scale)', fontsize=12, fontweight='bold')
    
    # Add current values
    latest = df.iloc[-1]
    info_text = (
        f"Current (Iter {int(latest['iteration'])}):\n"
        f"Total: {latest['total_loss']:.4f}\n"
        f"Value: {latest['value_loss']:.4f}\n"
        f"Policy: {latest['policy_loss']:.4f}"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8, edgecolor='black'))


def plot_game_stats(ax, df: pd.DataFrame):
    """Plot game length and buffer size."""
    ax.clear()
    
    if df.empty or len(df) < 2:
        ax.text(0.5, 0.5, 'Waiting for more training data...\n(Need at least 2 iterations)', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=13, color='#7f8c8d', style='italic')
        ax.set_title('Game Statistics', fontsize=15, fontweight='bold', pad=10)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Avg Game Length (moves)', fontsize=12)
        ax.grid(True, alpha=0.4, linestyle='--')
        return
    
    iterations = df['iteration']
    
    # Plot game length
    ax.plot(iterations, df['avg_game_length'], 
            label='Avg Game Length', color=COLORS['game_length'], 
            linewidth=3, marker='o', markersize=6)
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Game Length (moves)', fontsize=12, fontweight='bold', color=COLORS['game_length'])
    ax.tick_params(axis='y', labelcolor=COLORS['game_length'])
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.set_title('Game Statistics', fontsize=15, fontweight='bold', pad=10)
    
    # Add current value
    latest = df.iloc[-1]
    info_text = f"Current: {latest['avg_game_length']:.1f} moves/game"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8, edgecolor='black'))
    
    ax.legend(loc='best', framealpha=0.95, fontsize=11, edgecolor='black')


def plot_buffer_size(ax, df: pd.DataFrame):
    """Plot buffer size growth."""
    ax.clear()
    
    if df.empty or len(df) < 2 or 'buffer_size' not in df.columns:
        ax.text(0.5, 0.5, 'Waiting for more training data...\n(Need at least 2 iterations)', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=13, color='#7f8c8d', style='italic')
        ax.set_title('Replay Buffer Size', fontsize=15, fontweight='bold', pad=10)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Buffer Size (positions)', fontsize=12)
        ax.grid(True, alpha=0.4, linestyle='--')
        return
    
    iterations = df['iteration']
    
    # Plot buffer size
    ax.plot(iterations, df['buffer_size'], 
            label='Buffer Size', color=COLORS['buffer'], 
            linewidth=3, marker='s', markersize=6)
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Buffer Size (positions)', fontsize=12, fontweight='bold', color=COLORS['buffer'])
    ax.tick_params(axis='y', labelcolor=COLORS['buffer'])
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.set_title('Replay Buffer Size', fontsize=15, fontweight='bold', pad=10)
    
    # Format y-axis with commas
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add current value
    latest = df.iloc[-1]
    info_text = f"Current: {int(latest['buffer_size']):,} positions"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8, edgecolor='black'))
    
    ax.legend(loc='best', framealpha=0.95, fontsize=11, edgecolor='black')


# ════════════════════════════════════════════════════════════════════
# ANIMATION UPDATE
# ════════════════════════════════════════════════════════════════════

def update_plots(frame, filepath, axes):
    """Update all plots with latest data."""
    df = load_training_data(filepath)
    
    plot_win_rates(axes['win_rates'], df)
    plot_losses(axes['losses'], df)
    plot_game_stats(axes['game_stats'], df)
    plot_buffer_size(axes['buffer'], df)
    
    # Update title with timestamp and data info
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if not df.empty:
        title = f'AlphaZero Training Monitor - {current_time} | Iteration {len(df)}/{len(df)}'
    else:
        title = f'AlphaZero Training Monitor - {current_time} | Waiting for data...'
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    return []


# ════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ════════════════════════════════════════════════════════════════════

def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='AlphaZero Training Visualization Dashboard')
    parser.add_argument('--log-file', type=str, default=DEFAULT_LOG_FILE,
                       help='Path to CSV training log')
    parser.add_argument('--refresh-interval', type=int, default=30,
                       help='Refresh interval in seconds')
    parser.add_argument('--static', action='store_true',
                       help='Disable auto-refresh')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found: {args.log_file}")
        sys.exit(1)
    
    print("="*70)
    print("ALPHAZERO TRAINING VISUALIZATION DASHBOARD")
    print("="*70)
    print(f"Log file: {args.log_file}")
    print(f"Refresh interval: {args.refresh_interval}s")
    print(f"Auto-refresh: {'disabled' if args.static else 'enabled'}")
    print("="*70)
    print("\nOpening dashboard window...")
    print("Tip: Leave this running on a second monitor during training!")
    print("Press Ctrl+C to close\n")
    
    # Create figure with clean 2x2 grid
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('white')
    
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.30,
                  left=0.08, right=0.95, top=0.94, bottom=0.06)
    
    # Create axes in 2x2 grid
    ax_win_rates = fig.add_subplot(gs[0, :])      # Top row, full width
    ax_losses = fig.add_subplot(gs[1, 0])         # Bottom left
    ax_game_stats = fig.add_subplot(gs[1, 1])     # Bottom right top
    ax_buffer = fig.add_subplot(gs[1, 1])         # Share with game stats
    
    axes = {
        'win_rates': ax_win_rates,
        'losses': ax_losses,
        'game_stats': ax_game_stats,
        'buffer': ax_buffer,
    }
    
    # Initial plot
    update_plots(0, args.log_file, axes)
    
    if args.static:
        print("Static plot generated. Close window to exit.")
        plt.show()
    else:
        refresh_ms = args.refresh_interval * 1000
        ani = animation.FuncAnimation(
            fig, update_plots, fargs=(args.log_file, axes),
            interval=refresh_ms, cache_frame_data=False
        )
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\n\nDashboard closed by user")
            sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDashboard closed by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
