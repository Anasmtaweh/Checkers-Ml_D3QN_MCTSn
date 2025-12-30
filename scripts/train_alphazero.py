#!/usr/bin/env python3
"""
train_alphazero.py - AlphaZero Checkers Training Script

Trains an AlphaZero-style agent for checkers using:
- Self-play game generation with MCTS
- Neural network training (policy + value heads)
- Iterative improvement loop
"""

import torch
import torch.optim as optim
import numpy as np
import os
import csv
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from training.alpha_zero.network import AlphaZeroModel
from training.alpha_zero.mcts import MCTS
from training.alpha_zero.trainer import AlphaZeroTrainer
from core.action_manager import ActionManager
from core.board_encoder import CheckersBoardEncoder
from scripts.config_alphazero import CONFIGS, print_config

# ════════════════════════════════════════════════════════════════════
# SETTINGS
# ════════════════════════════════════════════════════════════════════

# Set to 0 to start fresh. Set to 4 (or your last iter) to resume.
RESUME_FROM_ITERATION = 0

# Paths
CHECKPOINT_DIR = "checkpoints/alphazero"
CSV_LOG_FILE = "data/training_logs/alphazero_training.csv"
BUFFER_PATH = os.path.join(CHECKPOINT_DIR, "latest_replay_buffer.pkl")

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_LOG_FILE), exist_ok=True)


# ════════════════════════════════════════════════════════════════════
# CSV LOGGING UTILITIES
# ════════════════════════════════════════════════════════════════════

def initialize_csv_log(filepath: str):
    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'timestamp', 'p1_wins', 'p2_wins', 'draws',
                'p1_win_rate', 'p2_win_rate', 'draw_rate', 'avg_game_length',
                'total_loss', 'value_loss', 'policy_loss', 'buffer_size', 'elapsed_time_s'
            ])
        print(f"✓ Created CSV log: {filepath}")

def log_to_csv(filepath: str, data: Dict[str, Any]):
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            data['iteration'], data['timestamp'], data['p1_wins'], data['p2_wins'],
            data['draws'], data['p1_win_rate'], data['p2_win_rate'], data['draw_rate'],
            data['avg_game_length'], data['total_loss'], data['value_loss'],
            data['policy_loss'], data['buffer_size'], data['elapsed_time']
        ])

# ════════════════════════════════════════════════════════════════════
# CHECKPOINT UTILITIES
# ════════════════════════════════════════════════════════════════════

def save_checkpoint(trainer, iteration, additional_info=None):
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_iter_{iteration}.pth")
    trainer.save_checkpoint(path, iteration, additional_info)

def load_checkpoint(trainer, iteration):
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_iter_{iteration}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return trainer.load_checkpoint(path)

# ════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ════════════════════════════════════════════════════════════════════

def main():
    # Parse CLI args
    parser = argparse.ArgumentParser(description='Train AlphaZero agent.')
    parser.add_argument('--config', type=str, default='standard', help='Configuration preset')
    parser.add_argument('--resume', type=int, default=None, help='Override resume iteration')
    args = parser.parse_args()

    if args.config not in CONFIGS:
        print(f"❌ Unknown configuration: {args.config}")
        return

    # Load Configuration
    CFG = CONFIGS[args.config]
    print_config(args.config)
    
    # Determine start iteration
    start_iter = args.resume if args.resume is not None else RESUME_FROM_ITERATION

    print("="*70)
    print("ALPHAZERO CHECKERS TRAINING")
    print("="*70)

    # Device Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Initialize Components
    print("\nInitializing components...")
    action_manager = ActionManager(device=device)
    encoder = CheckersBoardEncoder()
    model = AlphaZeroModel(action_dim=action_manager.action_dim, device=device)
    
    # MCTS
    mcts = MCTS(
        model=model,
        action_manager=action_manager,
        encoder=encoder,
        c_puct=1.5,
        num_simulations=CFG['MCTS_SIMULATIONS'], # Pulls 300 from Config
        device=device,
        dirichlet_alpha=0.6,
        dirichlet_epsilon=0.25
    )

    # Optimizer
    optimizer = optim.Adam(
        model.network.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    # Trainer
    trainer = AlphaZeroTrainer(
        model=model,
        mcts=mcts,
        action_manager=action_manager,
        board_encoder=encoder,
        optimizer=optimizer,
        device=device,
        buffer_size=CFG['BUFFER_SIZE'],  # Pulls 50000 from Config
        batch_size=CFG['BATCH_SIZE'],    # Pulls 512 from Config
        lr=0.001,
        weight_decay=1e-4,
        value_loss_weight=0.15,
        policy_loss_weight=1.0,
        temp_threshold=30,
    )

    print(f"  ✓ MCTS (simulations={mcts.num_simulations})")
    print(f"  ✓ AlphaZeroTrainer (buffer={CFG['BUFFER_SIZE']}, batch={CFG['BATCH_SIZE']})")

    # Resume Logic
    if start_iter > 0:
        print(f"\n🔄 Resuming training from iteration {start_iter}...")
        try:
            load_checkpoint(trainer, start_iter)
            trainer.load_replay_buffer(BUFFER_PATH)
            print(f"  ✓ Loaded checkpoint and buffer")
        except FileNotFoundError:
            print(f"  ❌ Checkpoint {start_iter} not found. Starting fresh.")
            start_iter = 0
    else:
        print("\n✨ Starting fresh training run")

    initialize_csv_log(CSV_LOG_FILE)

    # Training Loop
    for iteration in range(start_iter + 1, CFG['NUM_ITERATIONS'] + 1):
        iter_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}/{CFG['NUM_ITERATIONS']}")
        print(f"{'='*70}")

        # 1. Self Play
        print(f"\n[1/2] Self-Play ({CFG['GAMES_PER_ITERATION']} games)...")
        sp_stats = trainer.self_play(num_games=CFG['GAMES_PER_ITERATION'], verbose=True)

        # 2. Training
        print(f"\n[2/2] Training ({CFG['TRAIN_EPOCHS']} epochs)...")
        train_stats = trainer.train_step(epochs=CFG['TRAIN_EPOCHS'], verbose=True)

        # 3. Logging & Saving
        elapsed = time.time() - iter_start
        
        log_data = {
            'iteration': iteration,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'p1_wins': sp_stats['p1_wins'], 'p2_wins': sp_stats['p2_wins'],
            'draws': sp_stats['draws'], 'p1_win_rate': sp_stats['p1_win_rate'],
            'p2_win_rate': sp_stats['p2_win_rate'], 'draw_rate': sp_stats['draw_rate'],
            'avg_game_length': sp_stats['avg_game_length'],
            'total_loss': train_stats['loss'], 'value_loss': train_stats['value_loss'],
            'policy_loss': train_stats['policy_loss'],
            'buffer_size': sp_stats['buffer_size'], 'elapsed_time': elapsed
        }
        log_to_csv(CSV_LOG_FILE, log_data)
        
        # Save Checkpoint every time
        save_checkpoint(trainer, iteration, {'config': CFG})
        
        # Save Buffer periodically
        if iteration % 5 == 0:
            trainer.save_replay_buffer(BUFFER_PATH)

        print(f"\n✓ Iteration {iteration} complete in {elapsed:.1f}s")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted. Progress saved.")
        sys.exit(0)