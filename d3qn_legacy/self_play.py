#!/usr/bin/env python3
"""
self_play.py - Gen 12: Elite Refinement Training

Changes from Gen 10:
1. Starting from scratch (no corrupted checkpoints)
2. Draw penalty active from episode 1 (-0.5)
3. Progressive reward shaping for captures
4. No endgame injection (natural game flow)
5. Timeout = draw (not loss) with -0.5 penalty
6. New checkpoint directory: checkpoints_gen11_decisive

Author: ML Engineering Team
Date: December 21, 2025
"""

import torch
import torch.optim as optim
import numpy as np
import os
import glob
import random
import csv
import sys
import traceback
from collections import deque

# ════════════════════════════════════════════════════════════════════
# PROJECT PATH SETUP
# ════════════════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# ════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════

from checkers_env.env import CheckersEnv
from checkers_agents.random_agent import CheckersRandomAgent
from common.action_manager import ActionManager
from common.board_encoder import CheckersBoardEncoder
from common.buffer import ReplayBuffer
from d3qn_legacy.d3qn.model import D3QNModel
from d3qn_legacy.d3qn.trainer import D3QNTrainer

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

NUM_EPISODES = 15000           # Longer training for mastery
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.001
MAX_MOVES_PER_GAME = 800  # CHANGED from 500 (expert games need space)
NO_CAPTURE_LIMIT = 80     # NEW: 40 moves per player = standard rule

EPSILON_START = 0.50           # Higher initial exploration
EPSILON_END = 0.02             # Lower final exploration
EPSILON_DECAY = 4000           # Decay over more episodes

BUFFER_CAPACITY = 40000
MIN_BUFFER_SIZE = 5000

# NEW PATHS for Gen 11
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPPONENT_POOL_DIR = os.path.join(SCRIPT_DIR, "opponent_pool")
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints_gen12_elite")
LOG_FILE = os.path.join(SCRIPT_DIR, "gen12_elite_training.csv")

os.makedirs(OPPONENT_POOL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════
# UTILS
# ════════════════════════════════════════════════════════════════════

def log_print(msg):
    """Forces print to flush immediately."""
    print(msg, flush=True)

def get_learning_rate(episode):
    """5-phase learning rate schedule for Gen 12."""
    if episode < 500:
        return 2e-5      # Fast initial
    elif episode < 1500:
        return 5e-6      # Standard learning
    elif episode < 3000:
        return 1e-6      # Refinement
    elif episode < 6000:
        return 5e-7      # Fine-tuning
    else:
        return 1e-7      # Ultra-conservative

def get_opponent_probabilities(episode):
    """Gen 12: More self-play for positional mastery."""
    if episode < 1000:
        return 0.55, 0.20, 0.25  # Balanced early
    elif episode < 3000:
        return 0.50, 0.15, 0.35  # More self-play
    else:
        return 0.45, 0.15, 0.40  # Heavy self-play late

def calculate_epsilon(episode):
    """Epsilon-greedy exploration schedule."""
    epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) * (episode / EPSILON_DECAY)
    return max(EPSILON_END, epsilon)

def find_latest_checkpoint(checkpoint_dir):
    """Finds most recent checkpoint."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "gen12_elite_*.pth"))
    if not checkpoints:
        return None
    
    episodes = []
    for cp in checkpoints:
        try:
            ep = int(cp.split("_")[-1].replace(".pth", ""))
            episodes.append((ep, cp))
        except:
            continue
    
    if not episodes:
        return None
    
    return max(episodes, key=lambda x: x[0])[1]

def update_opponent_pool(checkpoint_dir, pool_dir, current_episode):
    """
    Update opponent pool with recent strong checkpoints.
    Keeps pool fresh and prevents outdated agent influence.
    
    Strategy:
    - Every 500 episodes, add latest checkpoint to pool
    - Keep only last 3 checkpoints (avoid pool bloat)
    - Remove oldest checkpoint when adding new one
    """
    if current_episode % 500 != 0 or current_episode < 1500:
        return  # Only update every 500 episodes, starting at 1500
    
    log_print(f"🔄 Updating opponent pool at episode {current_episode}...")
    
    # Get latest checkpoint
    latest_checkpoint = os.path.join(checkpoint_dir, f"gen12_elite_{current_episode}.pth")
    
    if not os.path.exists(latest_checkpoint):
        log_print(f"⚠️ Checkpoint {latest_checkpoint} not found, skipping pool update")
        return
    
    # Copy to pool with generation number
    pool_name = f"gen12_ep{current_episode}.pth"
    pool_path = os.path.join(pool_dir, pool_name)
    
    import shutil
    shutil.copy(latest_checkpoint, pool_path)
    log_print(f"✅ Added to pool: {pool_name}")
    
    # Keep only last 3 checkpoints in pool (clean old ones)
    pool_files = sorted(glob.glob(os.path.join(pool_dir, "gen12_ep*.pth")))
    
    if len(pool_files) > 3:
        # Remove oldest checkpoints
        for old_file in pool_files[:-3]:
            os.remove(old_file)
            log_print(f"🗑️ Removed old pool agent: {os.path.basename(old_file)}")
    
    # Show current pool
    remaining = [os.path.basename(f) for f in glob.glob(os.path.join(pool_dir, "*.pth"))]
    log_print(f"📋 Current pool ({len(remaining)} agents): {', '.join(remaining)}")

# ════════════════════════════════════════════════════════════════════
# CLASSES
# ════════════════════════════════════════════════════════════════════

class OpponentManager:
    """Manages opponent selection."""
    
    def __init__(self, pool_dir: str, action_manager, device):
        self.pool_dir = pool_dir
        self.action_manager = action_manager
        self.device = device
        self.opponent_model = D3QNModel(action_manager.action_dim, device=device).to(device)
        self.opponent_model.eval()
        
        log_print(f"🔍 Scanning for opponents in: {pool_dir}")
        
        all_files = glob.glob(os.path.join(pool_dir, "*.pth"))
        # Exclude gen7 and any corrupted gen10 checkpoints
        self.pool_files = [f for f in all_files if "gen7" not in f and "gen10_titan" not in f]
        
        if len(self.pool_files) == 0:
            log_print("⚠️ WARNING: Opponent pool is empty. Will use Random + Self-Play.")
        else:
            log_print(f"✅ Loaded Pool: {len(self.pool_files)} historical agents.")

    def get_opponent(self, current_online_model, prob_random, prob_pool):
        """Selects opponent based on probabilities."""
        if len(self.pool_files) == 0:
            prob_pool = 0.0
        
        rand = random.random()
        
        if rand < prob_random:
            return "Random", CheckersRandomAgent().select_action
        
        elif rand < prob_random + prob_pool:
            model_path = random.choice(self.pool_files)
            name = os.path.basename(model_path)
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                state_dict = checkpoint['model_online'] if isinstance(checkpoint, dict) and 'model_online' in checkpoint else checkpoint
                self.opponent_model.online.load_state_dict(state_dict)
                def pool_select(env): return self._model_select(self.opponent_model, env)
                return f"Pool: {name}", pool_select
            except:
                return "Random (Fallback)", CheckersRandomAgent().select_action
        
        else:
            def self_select(env): return self._model_select(current_online_model, env)
            return "Self-Play", self_select

    def _model_select(self, model, env):
        """Selects action using model Q-values."""
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            return None
        
        state_tensor = CheckersBoardEncoder().encode(env.board.get_state(), env.current_player).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q = model.get_q_values(state_tensor)[0]
        
        mask = self.action_manager.make_legal_action_mask(legal_moves).to(self.device)
        q[~mask] = -float('inf')
        
        action_id = int(q.argmax().item())
        move = self.action_manager.get_move_from_id(action_id)
        
        for lm in legal_moves:
            if isinstance(lm, list):
                if (tuple(lm[0][0]), tuple(lm[-1][1])) == move:
                    return lm
            elif len(lm) == 2:
                if (tuple(lm[0]), tuple(lm[1])) == move:
                    return lm
        
        return legal_moves[0]

def select_agent_action(model, state, legal_moves, action_manager, epsilon, device):
    """Epsilon-greedy action selection."""
    legal_mask = action_manager.make_legal_action_mask(legal_moves)
    legal_indices = torch.where(legal_mask)[0]
    
    if len(legal_indices) == 0:
        return None, None, legal_mask

    if np.random.random() < epsilon:
        action_id = int(legal_indices[np.random.randint(len(legal_indices))].item())
    else:
        model.eval()
        with torch.no_grad():
            q = model.get_q_values(state.unsqueeze(0).to(device))[0]
            q[~legal_mask] = -1e9
            action_id = int(q.argmax().item())

    move = action_manager.get_move_from_id(action_id)
    env_move = None
    
    for lm in legal_moves:
        if isinstance(lm, list) and (tuple(lm[0][0]), tuple(lm[-1][1])) == move:
            env_move = lm
            break
        elif len(lm) == 2 and (tuple(lm[0]), tuple(lm[1])) == move:
            env_move = lm
            break
            
    return (env_move if env_move else legal_moves[0]), action_id, legal_mask

# ════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ════════════════════════════════════════════════════════════════════

def main_training_loop():
    log_print("="*70)
    log_print("GEN 12: ELITE REFINEMENT TRAINING")
    log_print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = CheckersEnv()
    action_manager = ActionManager(device=device)
    encoder = CheckersBoardEncoder()
    
    # Initialize CSV log
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Episode', 'Side', 'Opponent', 'Result', 'Reward', 'Steps',
                'P1_WinRate', 'P2_WinRate', 'vsRandom_WinRate', 'Loss', 'Epsilon', 'LR', 'MaxQ', 'AvgQ'
            ])

    # Initialize model
    model = D3QNModel(action_manager.action_dim, device=device).to(device)
    
    # Check for resume
    resume_path = find_latest_checkpoint(CHECKPOINT_DIR)
    start_episode = 1
    
    if resume_path and os.path.exists(resume_path):
        log_print(f"🔄 RESUMING from Checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.online.load_state_dict(checkpoint['model_online'])
        model.target.load_state_dict(checkpoint['model_target'])
        start_episode = checkpoint['episode'] + 1
        log_print(f"⏩ Fast-Forwarding to Episode {start_episode}")
        optimizer = optim.Adam(model.online.parameters(), lr=get_learning_rate(start_episode))
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        log_print("🆕 Starting Fresh Training (No Resume Checkpoint)")
        optimizer = optim.Adam(model.online.parameters(), lr=get_learning_rate(1))

    # Initialize training components
    buffer = ReplayBuffer(BUFFER_CAPACITY, action_manager.action_dim, device)
    trainer = D3QNTrainer(env, action_manager, encoder, model, optimizer, buffer, device, 
                          gamma=GAMMA, tau=TAU, gradient_clip=0.1)  # CHANGED from 0.5
    opponent_manager = OpponentManager(OPPONENT_POOL_DIR, action_manager, device)

    # Training state tracking
    losses = []
    history_p1 = deque(maxlen=100)
    history_p2 = deque(maxlen=100)
    history_vs_random = deque(maxlen=100)
    
    log_print(f"🚀 Starting Training Loop...")

    # ════════════════════════════════════════════════════════════════════
    # EPISODE LOOP
    # ════════════════════════════════════════════════════════════════════

    for episode in range(start_episode, NUM_EPISODES + 1):
        
        # Update learning rate
        current_lr = get_learning_rate(episode)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        epsilon = calculate_epsilon(episode)
        agent_side = 1 if random.random() < 0.5 else -1
        
        # Always start with normal reset (no endgame injection)
        state = env.reset()
        p_rand, p_pool, p_self = get_opponent_probabilities(episode)

        done = False
        episode_reward = 0
        info = {}
        step_count = 0
        
        opp_name, opp_policy = opponent_manager.get_opponent(model, p_rand, p_pool)

        # Heartbeat
        if episode % 100 == 0:
            log_print(f"💓 Heartbeat: Episode {episode}...")

        # Opponent moves first if agent is P2
        if agent_side == -1:
            opp_action = opp_policy(env)
            if opp_action:
                state, _, done, _ = env.step(opp_action)
            else:
                done = True

        # Game loop
        move_history = []
        no_capture_count = 0
        while not done:
            step_count += 1
            
            # LENIENT repetition detection (only catches extreme stalling)
            if step_count > 150 and len(move_history) >= 30:
                recent_30 = move_history[-30:]
                unique_moves = len(set(recent_30))
                
                # Only trigger if EXTREMELY repetitive (2-3 moves repeated constantly)
                if unique_moves <= 3:
                    # Additional check: ensure it's truly back-and-forth
                    move_counts = {}
                    for move in recent_30:
                        move_counts[move] = move_counts.get(move, 0) + 1
                    
                    # Only punish if one move appears 10+ times in last 30 moves
                    if max(move_counts.values()) >= 10:
                        done = True
                        info['winner'] = -agent_side
                        log_print(f"  🔁 Repetition detected at step {step_count}, forcing loss")
                        break

            # Timeout check (existing)
            if step_count >= MAX_MOVES_PER_GAME:
                done = True
                info['winner'] = 0
                log_print(f"  ⏱️ Timeout at {MAX_MOVES_PER_GAME} moves (extreme endgame)")
                break

            current_player = env.current_player
            legal_moves = env.get_legal_moves()
            
            if not legal_moves:
                done = True
                break

            if current_player == agent_side:
                # Agent's turn
                encoded_state = encoder.encode(state, player=agent_side)
                env_move, action_id, _ = select_agent_action(model, encoded_state, legal_moves, action_manager, epsilon, device)
                
                if env_move is None or action_id is None:
                    done = True
                    break

                if env_move:
                    # Track move for repetition detection
                    if isinstance(env_move, list):
                        move_key = (tuple(env_move[0][0]), tuple(env_move[-1][1]))
                    else:
                        move_key = (tuple(env_move[0]), tuple(env_move[1]))
                    move_history.append(move_key)

                next_state, reward, done, info = env.step(env_move)
                
                # Track captures for 40-move rule
                if reward >= 0.001:  # Any capture occurred
                    no_capture_count = 0
                else:
                    no_capture_count += 1
                
                # Check 40-move rule (before timeout check)
                if no_capture_count >= NO_CAPTURE_LIMIT:
                    done = True
                    info['winner'] = 0
                    log_print(f"  📜 Draw by 40-move rule (no captures in {NO_CAPTURE_LIMIT} moves)")
                    custom_reward = -0.5  # Moderate draw penalty
                    
                    next_encoded = encoder.encode(next_state, player=agent_side)
                    next_mask = action_manager.make_legal_action_mask([])
                    buffer.push(encoded_state, action_id, custom_reward, next_encoded, True, next_mask)
                    break
                
                # REWARD SHAPING (Gen 12)
                winner = info.get('winner', 0)
                if done and winner == agent_side:
                    custom_reward = 1.0
                elif done and winner == -agent_side:
                    custom_reward = -1.0
                elif done:
                    custom_reward = -0.5  # Draw (keep moderate)
                else:
                    if reward >= 0.01:
                        custom_reward = 0.40  # Multi-jump (up from 0.30)
                    elif reward >= 0.001:
                        custom_reward = 0.20  # Single capture (up from 0.15)
                    elif reward > 0:
                        custom_reward = 0.04  # Minor gains (up from 0.03)
                    else:
                        custom_reward = -0.010  # Living tax (up from -0.008)
                
                next_encoded = encoder.encode(next_state, player=agent_side)
                next_mask = action_manager.make_legal_action_mask(env.get_legal_moves() if not done else [])
                
                buffer.push(encoded_state, action_id, custom_reward, next_encoded, done, next_mask)
                
                if len(buffer) > MIN_BUFFER_SIZE:
                    model.train()
                    losses.append(trainer.train_step(BATCH_SIZE))
                    trainer.update_target_network()

                episode_reward += custom_reward
                state = next_state
            
            else:
                # Opponent's turn
                opp_action = opp_policy(env)
                if opp_action:
                    state, _, done, info = env.step(opp_action)
                else:
                    done = True

        # Q-VALUE HEALTH MONITORING with AUTO-BRAKE
        max_q, avg_q = 0.0, 0.0
        
        if episode % 50 == 0 and len(buffer) > MIN_BUFFER_SIZE:
            with torch.no_grad():
                sample_size = min(1000, len(buffer))
                sample_states = torch.from_numpy(buffer.states[:sample_size]).to(device)
                sample_q = model.online(sample_states).abs()
                max_q = sample_q.max().item()
                avg_q = sample_q.mean().item()
                
                if episode % 100 == 0:
                    log_print(f"  [Q-Health] Max: {max_q:.2f} | Avg: {avg_q:.2f} | LR: {current_lr:.2e}")

                # AUTO-BRAKE: Q-values getting too high
                if max_q > 15.0 or avg_q > 5.0:
                    old_lr = optimizer.param_groups[0]['lr']
                    new_lr = max(old_lr * 0.5, 1e-8)  # Cut LR in half, floor at 1e-8
                    
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    
                    log_print(f"🚨 AUTO-BRAKE: Q-values high (Max: {max_q:.1f}, Avg: {avg_q:.1f})")
                    log_print(f"   Reducing LR: {old_lr:.2e} → {new_lr:.2e}")

                # EMERGENCY STOP: Complete meltdown
                if max_q > 30.0 or avg_q > 10.0:
                    log_print(f"🚨 CRITICAL FAILURE: Q-value explosion! Max: {max_q:.1f}, Avg: {avg_q:.1f}")
                    log_print(f"💾 Emergency save...")
                    torch.save({
                        'episode': episode,
                        'model_online': model.online.state_dict(),
                        'model_target': model.target.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, os.path.join(CHECKPOINT_DIR, f"gen12_elite_emergency_{episode}.pth"))
                    log_print("🛑 Training halted to prevent further damage.")
                    return

        # Clear CUDA cache
        if episode % 100 == 0 and device == "cuda":
            torch.cuda.empty_cache()
            log_print(f"  [Memory] CUDA cache cleared")

        # LOGGING
        winner = info.get('winner', 0)
        is_win = (winner == agent_side)
        
        if agent_side == 1:
            history_p1.append(1 if is_win else 0)
        else:
            history_p2.append(1 if is_win else 0)
        
        if "Random" in opp_name:
            history_vs_random.append(1 if is_win else 0)

        if episode % 10 == 0:
            avg_loss = np.mean(losses[-100:]) if losses else 0.0
            p1_wr = np.mean(history_p1) * 100 if history_p1 else 0.0
            p2_wr = np.mean(history_p2) * 100 if history_p2 else 0.0
            rand_wr = np.mean(history_vs_random) * 100 if history_vs_random else 0.0
            
            side_str = "Red (P1)" if agent_side == 1 else "Blk (P2)"
            res_str = "WIN " if is_win else "LOSS" if winner != 0 else "DRAW"
            
            log_print(f"Ep {episode:5d} | {side_str} | Vs {opp_name:25s} | {res_str} | Steps: {step_count:3d} | WR[P1]: {p1_wr:3.0f}% | WR[P2]: {p2_wr:3.0f}% | vsRand: {rand_wr:3.0f}%")
            
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode, side_str, opp_name, res_str, f"{episode_reward:.4f}", step_count,
                    f"{p1_wr:.1f}", f"{p2_wr:.1f}", f"{rand_wr:.1f}", 
                    f"{avg_loss:.5f}", f"{epsilon:.4f}", f"{current_lr:.2e}", f"{max_q:.2f}", f"{avg_q:.2f}"
                ])

        # CHECKPOINTING
        if episode % 500 == 0:
            path = os.path.join(CHECKPOINT_DIR, f"gen12_elite_{episode}.pth")
            torch.save({
                'episode': episode,
                'model_online': model.online.state_dict(),
                'model_target': model.target.state_dict(),
                'optimizer': optimizer.state_dict(),
                'buffer_len': len(buffer)
            }, path)
            log_print(f"💾 Checkpoint Saved: {path}")
            
            # NEW: Update opponent pool dynamically
            update_opponent_pool(CHECKPOINT_DIR, OPPONENT_POOL_DIR, episode)

def main():
    """Wrapper with error handling."""
    try:
        main_training_loop()
    except KeyboardInterrupt:
        log_print("\n⚠️ Training Interrupted by User (Ctrl+C)")
    except Exception as e:
        log_print(f"\n❌ FATAL CRASH: {e}")
        traceback.print_exc()
    finally:
        log_print("🛑 Script Execution Finished.")

if __name__ == "__main__":
    main()
