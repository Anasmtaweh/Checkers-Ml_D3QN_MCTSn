#!/usr/bin/env python3
"""
train_alphazero.py - Local Training Script (RTX 2060 Optimized)
"""

import torch
import torch.optim as optim
import numpy as np
import os
import csv
import sys
import time
import argparse
from datetime import datetime
from typing import Optional, Dict, Any

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

# Define Project Root
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

from mcts_workspace.training.alpha_zero.network import AlphaZeroModel
from mcts_workspace.training.alpha_zero.mcts import MCTS
from mcts_workspace.training.alpha_zero.trainer import AlphaZeroTrainer
from mcts_workspace.core.action_manager import ActionManager
from mcts_workspace.core.board_encoder import CheckersBoardEncoder
from mcts_workspace.scripts.config_alphazero import CONFIGS

# ════════════════════════════════════════════════════════════════════
# SETTINGS
# ════════════════════════════════════════════════════════════════════

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "alphazero")
CSV_LOG_FILE = os.path.join(PROJECT_ROOT, "data", "training_logs", "alphazero_training.csv")
BUFFER_PATH = os.path.join(CHECKPOINT_DIR, "latest_replay_buffer.pkl")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_LOG_FILE), exist_ok=True)

# ════════════════════════════════════════════════════════════════════
# CSV UTILS
# ════════════════════════════════════════════════════════════════════

def initialize_csv_log(filepath: str):
    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'timestamp', 
                'p1_wins', 'p2_wins', 'draws', 
                'p1_win_rate', 'p2_win_rate', 'draw_rate',
                'avg_game_length', 'buffer_size', 
                'total_loss', 'value_loss', 'policy_loss', 
                'elapsed_time_s'
            ])

def log_to_csv(filepath: str, data: Dict[str, Any]):
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            data['iteration'], data['timestamp'],
            data['p1_wins'], data['p2_wins'], data['draws'],
            data['p1_win_rate'], data['p2_win_rate'], data['draw_rate'],
            data['avg_game_length'], data['buffer_size'],
            data['total_loss'], data['value_loss'],
            data['policy_loss'], data['elapsed_time']
        ])

def save_checkpoint(trainer: AlphaZeroTrainer, iteration: int, info: Dict[str, Any]):
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_iter_{iteration}.pth")
    trainer.save_checkpoint(path, iteration, info)

# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='madras_local_resume')
    parser.add_argument('--resume', type=int, default=None)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    if args.config not in CONFIGS:
        print(f"❌ Config {args.config} not found.")
        return

    CFG = CONFIGS[args.config]
    print(f"🔥 Starting Local Training: {args.config}")
    print(f"   Workers: {args.workers}")
    if 'description' in CFG:
        print(f"   Description: {CFG['description']}")

    # 1. Device Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")

    # 2. Components
    action_manager = ActionManager(device=device)
    encoder = CheckersBoardEncoder()
    model = AlphaZeroModel(action_dim=action_manager.action_dim, device=device)

    # 3. MCTS (Initial Setup)
    initial_sims = CFG.get('MCTS_SIMULATIONS', 80)
    if 'phases' in CFG:
        initial_sims = CFG['phases'][0]['MCTS_SIMULATIONS']

    mcts = MCTS(
        model=model,
        action_manager=action_manager,
        encoder=encoder,
        c_puct=CFG.get('C_PUCT', 3.0),
        num_simulations=initial_sims,
        device=device,
        draw_value=CFG.get('MCTS_DRAW_VALUE', 0.0),
        # FIXED: Pass initial alpha from config
        dirichlet_alpha=CFG.get('DIRICHLET_ALPHA', 0.3),
        dirichlet_epsilon=CFG.get('DIRICHLET_EPSILON', 0.25)
    )

    # 4. Optimizer
    lr = CFG.get('LR', 0.001)
    optimizer = optim.Adam(model.network.parameters(), lr=lr, weight_decay=1e-4)

    # 5. Trainer
    trainer = AlphaZeroTrainer(
        model=model,
        mcts=mcts,
        action_manager=action_manager,
        board_encoder=encoder,
        optimizer=optimizer,
        device=device,
        buffer_size=CFG['BUFFER_SIZE'],
        batch_size=CFG['BATCH_SIZE'],
        num_ray_workers=args.workers,
        env_max_moves=CFG.get('ENV_MAX_MOVES', 200),
        draw_penalty=CFG.get('DRAW_PENALTY', 0.0),
        temp_threshold=CFG.get('TEMP_THRESHOLD', 30),
        dirichlet_epsilon=CFG.get('DIRICHLET_EPSILON', 0.25)
    )

    # 6. Resume Logic
    start_iter = 0
    if args.resume is not None:
        start_iter = args.resume
        print(f"🔄 Resuming from Iteration {start_iter}")
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_iter_{start_iter}.pth")
        
        try:
            trainer.load_checkpoint(ckpt_path)
            if os.path.exists(BUFFER_PATH):
                trainer.load_replay_buffer(BUFFER_PATH)
            else:
                print("⚠️  Warning: Replay buffer not found. Starting with empty buffer.")
        except FileNotFoundError:
            print(f"❌ Checkpoint {start_iter} not found. Starting Fresh.")
            start_iter = 0

    initialize_csv_log(CSV_LOG_FILE)

    # 7. Training Loop
    for iteration in range(start_iter + 1, CFG['NUM_ITERATIONS'] + 1):
        iter_start = time.time()
        
        # --- Handle Dynamic vs Static Config ---
        current_phase_name = "Static"
        if 'phases' in CFG:
            phase = CFG['phases'][0]
            for p in CFG['phases']:
                if p['iter_start'] <= iteration <= p['iter_end']:
                    phase = p
                    break
            current_phase_name = phase['name']
            
            # Update MCTS/Trainer (Phased)
            mcts.num_simulations = phase['MCTS_SIMULATIONS']
            mcts.dirichlet_epsilon = phase['DIRICHLET_EPSILON']
            mcts.dirichlet_alpha = phase.get('DIRICHLET_ALPHA', 0.3) # <--- ADDED
            mcts.search_draw_bias = phase.get('MCTS_SEARCH_DRAW_BIAS', 0.0)
            
            trainer.temp_threshold = phase['TEMP_THRESHOLD']
            trainer.env_max_moves = phase['ENV_MAX_MOVES']
            trainer.no_progress_plies = phase['NO_PROGRESS_PLIES']
        else:
            # Static Config (Madras Clone / Local Resume)
            current_phase_name = "Speed Run"
            
            mcts.num_simulations = CFG['MCTS_SIMULATIONS']
            mcts.dirichlet_epsilon = CFG['DIRICHLET_EPSILON']
            mcts.dirichlet_alpha = CFG.get('DIRICHLET_ALPHA', 0.3) # <--- ADDED: Critical for the 1.0 Hack
            mcts.search_draw_bias = CFG.get('MCTS_SEARCH_DRAW_BIAS', 0.0)
            
            trainer.temp_threshold = CFG['TEMP_THRESHOLD']
            trainer.env_max_moves = CFG['ENV_MAX_MOVES']
            trainer.no_progress_plies = CFG.get('NO_PROGRESS_PLIES', 80)

        print(f"\nITERATION {iteration} | {current_phase_name}")
        print(f"Sims: {mcts.num_simulations} | Bias: {mcts.search_draw_bias} | Alpha: {mcts.dirichlet_alpha}")

        # Run Self Play
        sp_stats = trainer.self_play(CFG['GAMES_PER_ITERATION'], verbose=True, iteration=iteration)
        
        # Run Training
        train_stats = trainer.train_step(epochs=CFG['TRAIN_EPOCHS'], verbose=True)
        
        # Logging
        elapsed = time.time() - iter_start
        
        log_data = {
            'iteration': iteration,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'p1_wins': sp_stats.get('p1_wins', 0), 
            'p2_wins': sp_stats.get('p2_wins', 0), 
            'draws': sp_stats.get('draws', 0),
            'p1_win_rate': sp_stats.get('p1_win_rate', 0.0),
            'p2_win_rate': sp_stats.get('p2_win_rate', 0.0),
            'draw_rate': sp_stats.get('draw_rate', 0.0),
            'avg_game_length': sp_stats.get('avg_game_length', 0),
            'buffer_size': sp_stats.get('buffer_size', 0),
            'total_loss': train_stats['loss'], 
            'value_loss': train_stats.get('value_loss', 0), 
            'policy_loss': train_stats.get('policy_loss', 0),
            'elapsed_time': elapsed
        }
        log_to_csv(CSV_LOG_FILE, log_data)
        
        # Print Summary
        print(f"  [Stats] P1: {log_data['p1_win_rate']:.1%} | P2: {log_data['p2_win_rate']:.1%} | Draw: {log_data['draw_rate']:.1%}")
        
        # Checkpoint
        save_checkpoint(trainer, iteration, {'config': CFG})
        if iteration % 5 == 0:
            trainer.save_replay_buffer(BUFFER_PATH)

        print(f"✓ Iteration {iteration} complete in {elapsed:.1f}s")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted. Progress saved.")
        sys.exit(0)