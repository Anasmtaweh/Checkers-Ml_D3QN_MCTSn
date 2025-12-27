#!/usr/bin/env python3
"""
plot_alphazero.py - AlphaZero Training Visualization Dashboard

Real-time monitoring dashboard for AlphaZero training progress.

Displays:
- Win rates over iterations (P1/P2/Draws)
- Training losses (Total/Value/Policy)
- Average game length trends
- Buffer size growth

Features:
- Auto-refreshes every 30 seconds
- Beautiful matplotlib styling
- Handles missing/incomplete data gracefully
- Can run alongside training script

Usage:
    python scripts/plot_alphazero.py [--log-file path/to/log.csv] [--refresh-interval 30]

Author: ML Engineer
Date: December 27, 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime


# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

DEFAULT_LOG_FILE = "data/training_logs/alphazero_training.csv"
REFRESH_INTERVAL = 30000  # milliseconds (30 seconds)

# Plot styling
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'p1': '#2E86AB',      # Blue
    'p2': '#A23B72',      # Purple
    'draws': '#F18F01',   # Orange
    'total_loss': '#C73E1D',     # Red
    'value_loss': '#6A994E',     # Green
    'policy_loss': '#BC4B51',    # Dark red
    'game_length': '#4A4E69',    # Gray-blue
    'buffer': '#9A8C98',         # Gray-purple
}


# ════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════

def load_training_data(filepath: str) -> pd.DataFrame:
    """
    Load training data from CSV log.
    
    Args:
        filepath: Path to CSV log file
    
    Returns:
        DataFrame with training statistics, or empty DataFrame if error
    """
    if not os.path.exists(filepath):
        print(f"⚠️  Log file not found: {filepath}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(filepath)
        
        # Validate required columns
        required_cols = [
            'iteration', 'p1_win_rate', 'p2_win_rate', 'draw_rate',
            'total_loss', 'value_loss', 'policy_loss', 'avg_game_length'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️  Missing columns in CSV: {missing_cols}")
            return pd.DataFrame()
        
        return df
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ════════════════════════════════════════════════════════════════════

def plot_win_rates(ax, df: pd.DataFrame):
    """
    Plot win rates over iterations.
    
    Args:
        ax: Matplotlib axis
        df: Training data DataFrame
    """
    ax.clear()
    
    if df.empty:
        ax.text(0.5, 0.5, 'No data available', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')
        ax.set_title('Win Rates Over Time', fontsize=14, fontweight='bold')
        return
    
    iterations = df['iteration']
    
    # Plot win rates
    ax.plot(iterations, df['p1_win_rate'] * 100, 
            label='P1 (Red)', color=COLORS['p1'], linewidth=2, marker='o', markersize=3)
    ax.plot(iterations, df['p2_win_rate'] * 100, 
            label='P2 (Black)', color=COLORS['p2'], linewidth=2, marker='s', markersize=3)
    ax.plot(iterations, df['draw_rate'] * 100, 
            label='Draws', color=COLORS['draws'], linewidth=2, marker='^', markersize=3)
    
    # Reference line at 50%
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Styling
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Win Rate (%)', fontsize=11)
    ax.set_title('Win Rates Over Time', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Add latest values as text
    if len(df) > 0:
        latest = df.iloc[-1]
        info_text = (
            f"Latest (Iter {int(latest['iteration'])}):\n"
            f"P1: {latest['p1_win_rate']*100:.1f}%  "
            f"P2: {latest['p2_win_rate']*100:.1f}%  "
            f"Draw: {latest['draw_rate']*100:.1f}%"
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_losses(ax, df: pd.DataFrame):
    """
    Plot training losses over iterations.
    
    Args:
        ax: Matplotlib axis
        df: Training data DataFrame
    """
    ax.clear()
    
    if df.empty:
        ax.text(0.5, 0.5, 'No data available', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')
        ax.set_title('Training Losses', fontsize=14, fontweight='bold')
        return
    
    iterations = df['iteration']
    
    # Plot losses
    ax.plot(iterations, df['total_loss'], 
            label='Total Loss', color=COLORS['total_loss'], linewidth=2.5, marker='o', markersize=3)
    ax.plot(iterations, df['value_loss'], 
            label='Value Loss', color=COLORS['value_loss'], linewidth=2, marker='s', markersize=3, alpha=0.8)
    ax.plot(iterations, df['policy_loss'], 
            label='Policy Loss', color=COLORS['policy_loss'], linewidth=2, marker='^', markersize=3, alpha=0.8)
    
    # Styling
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training Losses', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Use log scale if losses vary greatly
    if len(df) > 0:
        loss_range = df['total_loss'].max() - df['total_loss'].min()
        if loss_range > 10:
            ax.set_yscale('log')
    
    # Add latest values as text
    if len(df) > 0:
        latest = df.iloc[-1]
        info_text = (
            f"Latest (Iter {int(latest['iteration'])}):\n"
            f"Total: {latest['total_loss']:.4f}\n"
            f"Value: {latest['value_loss']:.4f}\n"
            f"Policy: {latest['policy_loss']:.4f}"
        )
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))


def plot_game_length_and_buffer(ax1, ax2, df: pd.DataFrame):
    """
    Plot average game length and buffer size.
    
    Args:
        ax1: Matplotlib axis for game length
        ax2: Matplotlib axis for buffer size (twin axis)
        df: Training data DataFrame
    """
    ax1.clear()
    ax2.clear()
    
    if df.empty:
        ax1.text(0.5, 0.5, 'No data available', 
                ha='center', va='center', transform=ax1.transAxes,
                fontsize=12, color='gray')
        ax1.set_title('Game Statistics', fontsize=14, fontweight='bold')
        return
    
    iterations = df['iteration']
    
    # Plot average game length on left axis
    line1 = ax1.plot(iterations, df['avg_game_length'], 
                     label='Avg Game Length', color=COLORS['game_length'], 
                     linewidth=2.5, marker='o', markersize=4)
    
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Average Game Length (moves)', fontsize=11, color=COLORS['game_length'])
    ax1.tick_params(axis='y', labelcolor=COLORS['game_length'])
    ax1.grid(True, alpha=0.3)
    
    # Plot buffer size on right axis (if available)
    if 'buffer_size' in df.columns:
        line2 = ax2.plot(iterations, df['buffer_size'], 
                         label='Buffer Size', color=COLORS['buffer'], 
                         linewidth=2, marker='s', markersize=3, linestyle='--', alpha=0.7)
        
        ax2.set_ylabel('Buffer Size (positions)', fontsize=11, color=COLORS['buffer'])
        ax2.tick_params(axis='y', labelcolor=COLORS['buffer'])
        
        # Format y-axis with thousands separator
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best', framealpha=0.9, fontsize=10)
    else:
        ax1.legend(loc='best', framealpha=0.9, fontsize=10)
    
    ax1.set_title('Game Statistics', fontsize=14, fontweight='bold', pad=15)
    
    # Add latest values as text
    if len(df) > 0:
        latest = df.iloc[-1]
        info_text = f"Latest: {latest['avg_game_length']:.1f} moves"
        if 'buffer_size' in df.columns:
            info_text += f"\nBuffer: {int(latest['buffer_size']):,}"
        
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))


def plot_training_summary(ax, df: pd.DataFrame, filepath: str):
    """
    Plot training summary statistics as text.
    
    Args:
        ax: Matplotlib axis
        df: Training data DataFrame
        filepath: Path to log file
    """
    ax.clear()
    ax.axis('off')
    
    if df.empty:
        summary_text = (
            "📊 AlphaZero Training Dashboard\n\n"
            "⚠️  No training data available yet\n\n"
            f"Monitoring: {filepath}\n"
            "Waiting for training to begin..."
        )
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
                fontsize=11, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        return
    
    # Calculate statistics
    total_iterations = len(df)
    latest = df.iloc[-1]
    
    # Moving averages for recent performance
    recent_window = min(10, len(df))
    recent_df = df.tail(recent_window)
    
    avg_p1_recent = recent_df['p1_win_rate'].mean() * 100
    avg_p2_recent = recent_df['p2_win_rate'].mean() * 100
    avg_draw_recent = recent_df['draw_rate'].mean() * 100
    
    # Trend indicators
    if len(df) >= 2:
        p1_trend = "↑" if df['p1_win_rate'].iloc[-1] > df['p1_win_rate'].iloc[-2] else "↓"
        loss_trend = "↓" if df['total_loss'].iloc[-1] < df['total_loss'].iloc[-2] else "↑"
    else:
        p1_trend = "—"
        loss_trend = "—"
    
    # Last update time
    if 'timestamp' in df.columns:
        last_update = df['timestamp'].iloc[-1]
    else:
        last_update = "N/A"
    
    # Build summary text
    summary_text = (
        "📊 AlphaZero Training Dashboard\n"
        "═" * 40 + "\n\n"
        f"🔄 Total Iterations: {total_iterations}\n"
        f"🕐 Last Update: {last_update}\n\n"
        "Recent Performance (last 10 iters):\n"
        f"  • P1 Win Rate: {avg_p1_recent:.1f}% {p1_trend}\n"
        f"  • P2 Win Rate: {avg_p2_recent:.1f}%\n"
        f"  • Draw Rate: {avg_draw_recent:.1f}%\n\n"
        f"Current Loss: {latest['total_loss']:.4f} {loss_trend}\n"
        f"Current Game Length: {latest['avg_game_length']:.1f} moves\n\n"
        f"📁 Log: {os.path.basename(filepath)}\n"
        "🔄 Auto-refresh: 30s"
    )
    
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
            fontsize=10, va='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))


# ════════════════════════════════════════════════════════════════════
# ANIMATION UPDATE
# ════════════════════════════════════════════════════════════════════

def update_plots(frame, filepath, axes):
    """
    Update all plots with latest data.
    
    Args:
        frame: Animation frame number (unused)
        filepath: Path to CSV log file
        axes: Dictionary of matplotlib axes
    
    Returns:
        Empty list (required by FuncAnimation)
    """
    # Load fresh data
    df = load_training_data(filepath)
    
    # Update each subplot
    plot_win_rates(axes['win_rates'], df)
    plot_losses(axes['losses'], df)
    plot_game_length_and_buffer(axes['game_length'], axes['buffer'], df)
    plot_training_summary(axes['summary'], df, filepath)
    
    # Update main title with timestamp
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    plt.suptitle(f'AlphaZero Training Monitor — {current_time}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    return []  # Return empty list to satisfy matplotlib.animation


# ════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ════════════════════════════════════════════════════════════════════

def main():
    """Main visualization function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='AlphaZero Training Visualization Dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/plot_alphazero.py
  python scripts/plot_alphazero.py --log-file custom_log.csv
  python scripts/plot_alphazero.py --refresh-interval 60
        """
    )
    parser.add_argument('--log-file', type=str, default=DEFAULT_LOG_FILE,
                       help='Path to CSV training log (default: %(default)s)')
    parser.add_argument('--refresh-interval', type=int, default=30,
                       help='Refresh interval in seconds (default: %(default)s)')
    parser.add_argument('--static', action='store_true',
                       help='Disable auto-refresh (plot once and close)')
    
    args = parser.parse_args()
    
    # Validate log file
    if not os.path.exists(args.log_file):
        print(f"❌ Log file not found: {args.log_file}")
        print(f"   Make sure training has started and created the log file.")
        sys.exit(1)
    
    print("="*70)
    print("ALPHAZERO TRAINING VISUALIZATION DASHBOARD")
    print("="*70)
    print(f"Log file: {args.log_file}")
    print(f"Refresh interval: {args.refresh_interval}s")
    print(f"Auto-refresh: {'disabled' if args.static else 'enabled'}")
    print("="*70)
    print("\n📊 Opening dashboard window...")
    print("💡 Tip: Leave this running on a second monitor during training!")
    print("🔄 Press Ctrl+C to close\n")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.08, right=0.95, top=0.93, bottom=0.06)
    
    # Create axes
    ax_win_rates = fig.add_subplot(gs[0, :])  # Top row, full width
    ax_losses = fig.add_subplot(gs[1, 0])     # Middle left
    ax_game_length = fig.add_subplot(gs[1, 1])  # Middle right
    ax_buffer = ax_game_length.twinx()        # Twin axis for buffer size
    ax_summary = fig.add_subplot(gs[2, :])    # Bottom row, full width
    
    axes = {
        'win_rates': ax_win_rates,
        'losses': ax_losses,
        'game_length': ax_game_length,
        'buffer': ax_buffer,
        'summary': ax_summary,
    }
    
    # Initial plot
    update_plots(0, args.log_file, axes)
    
    if args.static:
        # Static mode: just show the plot once
        print("📊 Static plot generated. Close window to exit.")
        plt.show()
    else:
        # Animated mode: update every N seconds
        refresh_ms = args.refresh_interval * 1000
        ani = animation.FuncAnimation(
            fig, 
            update_plots, 
            fargs=(args.log_file, axes),
            interval=refresh_ms,
            cache_frame_data=False
        )
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\n\n⚠️  Dashboard closed by user")
            sys.exit(0)


# ════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Dashboard closed by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
