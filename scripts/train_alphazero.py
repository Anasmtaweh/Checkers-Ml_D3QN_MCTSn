#!/usr/bin/env python3
"""
train_alphazero.py - AlphaZero Checkers Training Script

Trains an AlphaZero-style agent for checkers using:
- Self-play game generation with MCTS
- Neural network training (policy + value heads)
- Iterative improvement loop

Key Features:
- CSV logging for tracking training progress
- Checkpoint save/load with resume capability
- Configurable hyperparameters
- Progress monitoring and statistics

Author: ML Engineer
Date: December 27, 2025
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
# HYPERPARAMETERS / CONFIGURATION
# ════════════════════════════════════════════════════════════════════

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train AlphaZero agent.')
parser.add_argument('--config', type=str, default='standard',
                    help=f'Configuration preset (default: standard). Options: {", ".join(CONFIGS.keys())}')
args = parser.parse_args()

# Select configuration
if args.config not in CONFIGS:
    print(f"❌ Unknown configuration: {args.config}")
    print(f"Available: {', '.join(CONFIGS.keys())}")
    sys.exit(1)

ACTIVE_CONFIG = CONFIGS[args.config]
print_config(args.config)

# Training Loop
NUM_ITERATIONS = ACTIVE_CONFIG['NUM_ITERATIONS']
GAMES_PER_ITERATION = ACTIVE_CONFIG['GAMES_PER_ITERATION']
TRAIN_EPOCHS = ACTIVE_CONFIG['TRAIN_EPOCHS']

# MCTS Parameters
MCTS_SIMULATIONS = ACTIVE_CONFIG['MCTS_SIMULATIONS']
MCTS_C_PUCT = 1.5              # Exploration constant (1.0-3.0 typical)
MCTS_TEMP_THRESHOLD = 30       # Move number to switch from temp=1.0 to temp=0.0

# Neural Network
LEARNING_RATE = 0.001          # Adam learning rate
WEIGHT_DECAY = 1e-4            # L2 regularization
BATCH_SIZE = ACTIVE_CONFIG['BATCH_SIZE']
BUFFER_SIZE = ACTIVE_CONFIG['BUFFER_SIZE']

# Loss Weights
VALUE_LOSS_WEIGHT = 0.3        # Weight for value loss
POLICY_LOSS_WEIGHT = 1.0       # Weight for policy loss

# Logging and Checkpointing
LOG_FREQ = 1                   # Log every N iterations
SAVE_FREQ = 5                  # Save checkpoint every N iterations
CSV_LOG_FILE = "data/training_logs/alphazero_training.csv"

# Paths
CHECKPOINT_DIR = "checkpoints/alphazero"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_LOG_FILE), exist_ok=True)

# Resume Configuration
# Set to iteration number to resume from checkpoint, or None to start fresh
# Example: 10 will load checkpoint_iter_10.pth and continue from iteration 11
RESUME_FROM_ITERATION: Optional[int] = None


# ════════════════════════════════════════════════════════════════════
# CSV LOGGING UTILITIES
# ════════════════════════════════════════════════════════════════════

def initialize_csv_log(filepath: str):
    """
    Initialize CSV log file with headers.
    
    Args:
        filepath: Path to CSV file
    """
    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration',
                'timestamp',
                'p1_wins',
                'p2_wins',
                'draws',
                'p1_win_rate',
                'p2_win_rate',
                'draw_rate',
                'avg_game_length',
                'total_loss',
                'value_loss',
                'policy_loss',
                'buffer_size',
                'elapsed_time_s'
            ])
        print(f"✓ Created CSV log: {filepath}")


def log_to_csv(filepath: str, data: Dict[str, Any]):
    """
    Append training statistics to CSV log.
    
    Args:
        filepath: Path to CSV file
        data: Dictionary with statistics to log
    """
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            data['iteration'],
            data['timestamp'],
            data['p1_wins'],
            data['p2_wins'],
            data['draws'],
            data['p1_win_rate'],
            data['p2_win_rate'],
            data['draw_rate'],
            data['avg_game_length'],
            data['total_loss'],
            data['value_loss'],
            data['policy_loss'],
            data['buffer_size'],
            data['elapsed_time']
        ])


# ════════════════════════════════════════════════════════════════════
# CHECKPOINT UTILITIES
# ════════════════════════════════════════════════════════════════════

def save_checkpoint(
    trainer: AlphaZeroTrainer,
    iteration: int,
    checkpoint_dir: str,
    additional_info: Optional[Dict] = None
):
    """
    Save training checkpoint.
    
    Args:
        trainer: AlphaZeroTrainer instance
        iteration: Current iteration number
        checkpoint_dir: Directory to save checkpoint
        additional_info: Optional additional information to save
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{iteration}.pth")
    trainer.save_checkpoint(checkpoint_path, iteration, additional_info)


def load_checkpoint(
    trainer: AlphaZeroTrainer,
    iteration: int,
    checkpoint_dir: str
) -> Dict:
    """
    Load training checkpoint.
    
    Args:
        trainer: AlphaZeroTrainer instance
        iteration: Iteration number to load
        checkpoint_dir: Directory containing checkpoint
    
    Returns:
        Checkpoint dictionary
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{iteration}.pth")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return trainer.load_checkpoint(checkpoint_path)


# ════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ════════════════════════════════════════════════════════════════════

def main():
    """Main training loop."""
    
    print("="*70)
    print("ALPHAZERO CHECKERS TRAINING")
    print("="*70)
    
    # ════════════════════════════════════════════════════════════════
    # DEVICE DETECTION
    # ════════════════════════════════════════════════════════════════
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # ════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ════════════════════════════════════════════════════════════════
    print("\nInitializing components...")
    
    # Action Manager
    action_manager = ActionManager(device=device)
    print(f"  ✓ ActionManager ({action_manager.action_dim} actions)")
    
    # Board Encoder
    encoder = CheckersBoardEncoder()
    print("  ✓ CheckersBoardEncoder")
    
    # AlphaZero Model
    model = AlphaZeroModel(
        action_dim=action_manager.action_dim,
        device=device
    )
    print(f"  ✓ AlphaZeroModel ({model.num_parameters():,} parameters)")
    
    # MCTS
    mcts = MCTS(
        model=model,
        action_manager=action_manager,
        encoder=encoder,
        c_puct=MCTS_C_PUCT,
        num_simulations=MCTS_SIMULATIONS,
        device=device,
        dirichlet_alpha=0.6,
        dirichlet_epsilon=0.25
    )
    print(f"  ✓ MCTS (simulations={MCTS_SIMULATIONS}, c_puct={MCTS_C_PUCT})")
    
    # Optimizer
    optimizer = optim.Adam(
        model.network.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    print(f"  ✓ Adam Optimizer (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    
    # Trainer
    trainer = AlphaZeroTrainer(
        model=model,
        mcts=mcts,
        action_manager=action_manager,
        board_encoder=encoder,
        optimizer=optimizer,
        device=device,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        value_loss_weight=VALUE_LOSS_WEIGHT,
        policy_loss_weight=POLICY_LOSS_WEIGHT,
        temp_threshold=MCTS_TEMP_THRESHOLD,
    )
    print(f"  ✓ AlphaZeroTrainer (buffer={BUFFER_SIZE}, batch={BATCH_SIZE})")
    
    # ════════════════════════════════════════════════════════════════
    # RESUME FROM CHECKPOINT (if specified)
    # ════════════════════════════════════════════════════════════════
    start_iteration = 1
    
    if RESUME_FROM_ITERATION is not None:
        print(f"\n🔄 Resuming training from iteration {RESUME_FROM_ITERATION}...")
        
        try:
            checkpoint = load_checkpoint(trainer, RESUME_FROM_ITERATION, CHECKPOINT_DIR)
            start_iteration = RESUME_FROM_ITERATION + 1
            
            print(f"  ✓ Successfully loaded checkpoint")
            print(f"  ✓ Resuming from iteration {start_iteration}")
            
            # Display checkpoint info
            if 'self_play_stats' in checkpoint:
                sp_stats = checkpoint['self_play_stats']
                print(f"  ✓ Previous P1 win rate: {sp_stats.get('p1_win_rate', 0):.1%}")
            
        except FileNotFoundError as e:
            print(f"  ❌ {e}")
            print(f"  ⚠️  Starting fresh training from iteration 1")
            start_iteration = 1
        except Exception as e:
            print(f"  ❌ Error loading checkpoint: {e}")
            print(f"  ⚠️  Starting fresh training from iteration 1")
            start_iteration = 1
    else:
        print("\n✨ Starting fresh training run")
    
    # ════════════════════════════════════════════════════════════════
    # INITIALIZE CSV LOG
    # ════════════════════════════════════════════════════════════════
    initialize_csv_log(CSV_LOG_FILE)
    
    # ════════════════════════════════════════════════════════════════
    # PRINT CONFIGURATION
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Iterations: {start_iteration} → {NUM_ITERATIONS}")
    print(f"Games per iteration: {GAMES_PER_ITERATION}")
    print(f"Training epochs: {TRAIN_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Buffer size: {BUFFER_SIZE}")
    print(f"\nMCTS Configuration:")
    print(f"  Simulations per move: {MCTS_SIMULATIONS}")
    print(f"  Exploration constant (c_puct): {MCTS_C_PUCT}")
    print(f"  Temperature threshold: {MCTS_TEMP_THRESHOLD} moves")
    print(f"\nNeural Network:")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Value loss weight: {VALUE_LOSS_WEIGHT}")
    print(f"  Policy loss weight: {POLICY_LOSS_WEIGHT}")
    print(f"\nLogging:")
    print(f"  CSV log: {CSV_LOG_FILE}")
    print(f"  Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"  Save frequency: every {SAVE_FREQ} iterations")
    print("="*70 + "\n")
    
    # ════════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ════════════════════════════════════════════════════════════════
    
    training_start_time = time.time()
    
    for iteration in range(start_iteration, NUM_ITERATIONS + 1):
        iteration_start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}/{NUM_ITERATIONS}")
        print(f"{'='*70}")
        
        # ════════════════════════════════════════════════════════════
        # PHASE 1: SELF-PLAY
        # ════════════════════════════════════════════════════════════
        print(f"\n[1/2] Self-Play ({GAMES_PER_ITERATION} games)...")
        self_play_start = time.time()
        
        self_play_stats = trainer.self_play(
            num_games=GAMES_PER_ITERATION,
            verbose=True
        )
        
        self_play_time = time.time() - self_play_start
        print(f"  Self-play completed in {self_play_time:.1f}s")
        
        # ════════════════════════════════════════════════════════════
        # PHASE 2: TRAINING
        # ════════════════════════════════════════════════════════════
        print(f"\n[2/2] Training ({TRAIN_EPOCHS} epochs)...")
        train_start = time.time()
        
        train_stats = trainer.train_step(
            epochs=TRAIN_EPOCHS,
            verbose=True
        )
        
        train_time = time.time() - train_start
        print(f"  Training completed in {train_time:.1f}s")
        
        # ════════════════════════════════════════════════════════════
        # LOGGING TO CSV
        # ════════════════════════════════════════════════════════════
        iteration_time = time.time() - iteration_start_time
        
        log_data = {
            'iteration': iteration,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'p1_wins': self_play_stats['p1_wins'],
            'p2_wins': self_play_stats['p2_wins'],
            'draws': self_play_stats['draws'],
            'p1_win_rate': self_play_stats['p1_win_rate'],
            'p2_win_rate': self_play_stats['p2_win_rate'],
            'draw_rate': self_play_stats['draw_rate'],
            'avg_game_length': self_play_stats['avg_game_length'],
            'total_loss': train_stats['loss'],
            'value_loss': train_stats['value_loss'],
            'policy_loss': train_stats['policy_loss'],
            'buffer_size': self_play_stats['buffer_size'],
            'elapsed_time': iteration_time
        }
        
        log_to_csv(CSV_LOG_FILE, log_data)
        
        # ════════════════════════════════════════════════════════════
        # CONSOLE SUMMARY
        # ════════════════════════════════════════════════════════════
        if iteration % LOG_FREQ == 0:
            print(f"\n{'─'*70}")
            print(f"ITERATION {iteration} SUMMARY")
            print(f"{'─'*70}")
            print(f"Self-Play Results:")
            print(f"  P1 Wins: {self_play_stats['p1_wins']:3d} ({self_play_stats['p1_win_rate']:6.1%})")
            print(f"  P2 Wins: {self_play_stats['p2_wins']:3d} ({self_play_stats['p2_win_rate']:6.1%})")
            print(f"  Draws:   {self_play_stats['draws']:3d} ({self_play_stats['draw_rate']:6.1%})")
            print(f"  Avg Length: {self_play_stats['avg_game_length']:.1f} moves")
            print(f"\nTraining Loss:")
            print(f"  Total:  {train_stats['loss']:.4f}")
            print(f"  Value:  {train_stats['value_loss']:.4f}")
            print(f"  Policy: {train_stats['policy_loss']:.4f}")
            print(f"\nBuffer Size: {self_play_stats['buffer_size']:,} positions")
            print(f"Iteration Time: {iteration_time:.1f}s")
            print(f"{'─'*70}")
        
        # ════════════════════════════════════════════════════════════
        # CHECKPOINT SAVING
        # ════════════════════════════════════════════════════════════
        if iteration % SAVE_FREQ == 0:
            save_checkpoint(
                trainer=trainer,
                iteration=iteration,
                checkpoint_dir=CHECKPOINT_DIR,
                additional_info={
                    'self_play_stats': self_play_stats,
                    'train_stats': train_stats,
                    'config': {
                        'games_per_iteration': GAMES_PER_ITERATION,
                        'train_epochs': TRAIN_EPOCHS,
                        'mcts_simulations': MCTS_SIMULATIONS,
                        'mcts_c_puct': MCTS_C_PUCT,
                        'learning_rate': LEARNING_RATE,
                        'batch_size': BATCH_SIZE,
                    }
                }
            )
        
        print(f"\n✓ Iteration {iteration} complete")
    
    # ════════════════════════════════════════════════════════════════
    # TRAINING COMPLETE
    # ════════════════════════════════════════════════════════════════
    total_training_time = time.time() - training_start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    # Save final model
    final_checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint_final.pth")
    trainer.save_checkpoint(
        final_checkpoint_path,
        iteration=NUM_ITERATIONS,
        additional_info={'final_model': True}
    )
    
    # Print final statistics
    print(f"\nTraining Statistics:")
    print(f"  Total iterations: {NUM_ITERATIONS}")
    print(f"  Total games played: {trainer.training_stats['total_games']}")
    print(f"  Total training time: {total_training_time/3600:.2f} hours")
    print(f"  Avg time per iteration: {total_training_time/NUM_ITERATIONS:.1f}s")
    print(f"\nFinal buffer size: {trainer.get_buffer_size():,} positions")
    print(f"\nCheckpoints saved in: {CHECKPOINT_DIR}")
    print(f"Training log saved in: {CSV_LOG_FILE}")
    print("="*70)


# ════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    print("\nStarting AlphaZero Checkers Training...")
    print("Press Ctrl+C to interrupt training\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Progress has been saved in checkpoints/alphazero/")
        print("Resume by setting RESUME_FROM_ITERATION in the script")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        import ray
        if ray.is_initialized():
            ray.shutdown()
