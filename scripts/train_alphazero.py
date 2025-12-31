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
                'config_name',
                'mcts_simulations',
                'env_max_moves',
                'no_progress_plies',
                'draw_penalty',
                'mcts_draw_value',
                'iteration', 'timestamp', 'p1_wins', 'p2_wins', 'draws',
                'p1_win_rate', 'p2_win_rate', 'draw_rate', 'avg_game_length',
                'move_mapping_failures',
                'total_loss', 'value_loss', 'policy_loss', 'buffer_size', 'elapsed_time_s'
            ])
        print(f"✓ Created CSV log: {filepath}")

def log_to_csv(filepath: str, data: Dict[str, Any]):
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            data['config_name'],
            data['mcts_simulations'],
            data['env_max_moves'],
            data['no_progress_plies'],
            data['draw_penalty'],
            data['mcts_draw_value'],
            data['iteration'], data['timestamp'], data['p1_wins'], data['p2_wins'],
            data['draws'], data['p1_win_rate'], data['p2_win_rate'], data['draw_rate'],
            data['avg_game_length'], data['move_mapping_failures'],
            data['total_loss'], data['value_loss'],
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

    # Get base config (handle phased curriculum)
    if 'phases' in CFG:
        first_phase = CFG['phases'][0]
        env_max_moves = first_phase.get('ENV_MAX_MOVES', 200)
        no_progress_plies = first_phase.get('NO_PROGRESS_PLIES', 80)
        draw_penalty = first_phase.get('DRAW_PENALTY', -0.1)
        mcts_draw_value = first_phase.get('MCTS_DRAW_VALUE', draw_penalty)
        dirichlet_epsilon = first_phase.get('DIRICHLET_EPSILON', 0.1)
        temp_threshold = first_phase.get('TEMP_THRESHOLD', 20)
        mcts_simulations = first_phase.get('MCTS_SIMULATIONS', 400)
        search_draw_bias = first_phase.get('MCTS_SEARCH_DRAW_BIAS', -0.03)
        print(f"  Initialized with Phase 1 settings: sims={mcts_simulations}, draw_bias={search_draw_bias}")
    else:
        env_max_moves = CFG.get('ENV_MAX_MOVES', 200)
        no_progress_plies = CFG.get('NO_PROGRESS_PLIES', 80)
        draw_penalty = CFG.get('DRAW_PENALTY', -0.1)
        mcts_draw_value = CFG.get('MCTS_DRAW_VALUE', draw_penalty)
        dirichlet_epsilon = CFG.get('DIRICHLET_EPSILON', 0.1)
        temp_threshold = CFG.get('TEMP_THRESHOLD', 20)
        mcts_simulations = CFG.get('MCTS_SIMULATIONS', 400)
        search_draw_bias = CFG.get('MCTS_SEARCH_DRAW_BIAS', -0.03)

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
        num_simulations=mcts_simulations, # Pulls from extracted config
        device=device,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.1,
        draw_value=mcts_draw_value,
    )

    # Optimizer
    optimizer = optim.Adam(
        model.network.parameters(),
        lr=0.001,
        weight_decay=1e-3
    )

    # Trainer
    trainer = AlphaZeroTrainer(
        model=model,
        mcts=mcts,
        action_manager=action_manager,
        board_encoder=encoder,
        optimizer=optimizer,
        device=device,
        buffer_size=CFG['BUFFER_SIZE'],  # Pulls 5000 from Config (updated)
        batch_size=CFG['BATCH_SIZE'],    # Pulls 256 from Config (updated)
        lr=0.001,
        weight_decay=1e-3,
        value_loss_weight=1.0,
        policy_loss_weight=1.0,
        temp_threshold=20,
        draw_penalty=draw_penalty,
        env_max_moves=env_max_moves,
        no_progress_plies=no_progress_plies,
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

        # Apply phase-specific parameters if using phased curriculum
        if 'phases' in CFG:
            phase_cfg = None
            for phase in CFG['phases']:
                if phase['iter_start'] <= iteration <= phase['iter_end']:
                    phase_cfg = phase
                    break
            
            if phase_cfg:
                print(f"\n📋 {phase_cfg['name']}")
                # Update MCTS parameters
                mcts.num_simulations = phase_cfg.get('MCTS_SIMULATIONS', mcts.num_simulations)
                mcts.dirichlet_epsilon = phase_cfg.get('DIRICHLET_EPSILON', mcts.dirichlet_epsilon)
                mcts.search_draw_bias = phase_cfg.get('MCTS_SEARCH_DRAW_BIAS', mcts.search_draw_bias)
                
                # Update trainer parameters
                trainer.temp_threshold = phase_cfg.get('TEMP_THRESHOLD', trainer.temp_threshold)
                trainer.env_max_moves = phase_cfg.get('ENV_MAX_MOVES', trainer.env_max_moves)
                trainer.no_progress_plies = phase_cfg.get('NO_PROGRESS_PLIES', trainer.no_progress_plies)
                trainer.draw_penalty = phase_cfg.get('DRAW_PENALTY', trainer.draw_penalty)
                mcts.draw_value = phase_cfg.get('MCTS_DRAW_VALUE', mcts.draw_value)
                
                print(f"  MCTS sims: {mcts.num_simulations}, Dirichlet ε: {mcts.dirichlet_epsilon}, Search bias: {mcts.search_draw_bias}")
                print(f"  Temp threshold: {trainer.temp_threshold}, Max moves: {trainer.env_max_moves}, No-progress: {trainer.no_progress_plies}")
                
                # ==================== DIAGNOSTIC 3: Draw Parameter Consistency ====================
                print(f"  [Draw Handling] Trainer penalty={trainer.draw_penalty:.3f}, MCTS value={mcts.draw_value:.3f}")
                
                # Critical assertion: these must match to avoid train/inference mismatch
                if abs(trainer.draw_penalty - mcts.draw_value) > 0.001:
                    print(f"  ⚠️  WARNING: Draw penalty mismatch! Trainer={trainer.draw_penalty:.3f}, MCTS={mcts.draw_value:.3f}")
                    print(f"       This will cause the network to learn different values than MCTS evaluates.")
                else:
                    print(f"  ✓ Draw parameters consistent")
                # ==================================================================================

        # 1. Self Play
        print(f"\n[1/2] Self-Play ({CFG['GAMES_PER_ITERATION']} games)...")
        sp_stats = trainer.self_play(num_games=CFG['GAMES_PER_ITERATION'], verbose=True, iteration=iteration)

        # 2. Training
        print(f"\n[2/2] Training ({CFG['TRAIN_EPOCHS']} epochs)...")
        train_stats = trainer.train_step(epochs=CFG['TRAIN_EPOCHS'], verbose=True)

        # 3. Logging & Saving
        elapsed = time.time() - iter_start
        
        log_data = {
            'config_name': args.config,
            'mcts_simulations': mcts.num_simulations,
            'env_max_moves': trainer.env_max_moves,
            'no_progress_plies': trainer.no_progress_plies,
            'draw_penalty': trainer.draw_penalty,
            'mcts_draw_value': mcts.draw_value,
            'iteration': iteration,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'p1_wins': sp_stats['p1_wins'], 'p2_wins': sp_stats['p2_wins'],
            'draws': sp_stats['draws'], 'p1_win_rate': sp_stats['p1_win_rate'],
            'p2_win_rate': sp_stats['p2_win_rate'], 'draw_rate': sp_stats['draw_rate'],
            'avg_game_length': sp_stats['avg_game_length'],
            'move_mapping_failures': sp_stats.get('move_mapping_failures', 0),
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