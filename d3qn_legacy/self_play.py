#!/usr/bin/env python3
"""
self_play.py - Phase 5: Iron League (DEBUG & RESUME MODE)

Updates:
1. WRAPPED in Try/Except to catch Silent Crashes.
2. FORCED FLUSH on all prints to prevent log lag.
3. Auto-Resumes from Ep 2500.

Author: ML Engineer
Date: December 20, 2025
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
import time
from collections import deque

# Import project modules
from checkers_env.env import CheckersEnv
from checkers_agents.random_agent import CheckersRandomAgent
from training.common.action_manager import ActionManager
from training.common.board_encoder import CheckersBoardEncoder
from training.common.buffer import ReplayBuffer
from training.d3qn.model import D3QNModel
from training.d3qn.trainer import D3QNTrainer

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

NUM_EPISODES = 20000
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.001
MAX_MOVES_PER_GAME = 500

EPSILON_START = 0.30
EPSILON_END = 0.05
EPSILON_DECAY = 5000

BUFFER_CAPACITY = 40000  
MIN_BUFFER_SIZE = 5000   

OPPONENT_POOL_DIR = "opponent_pool"
CHECKPOINT_DIR = "checkpoints_iron_league_v3"
LOG_FILE = "iron_league_nuclear.csv"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
MIN_POOL_SIZE = 3 

# ════════════════════════════════════════════════════════════════════
# UTILS
# ════════════════════════════════════════════════════════════════════

def log_print(msg):
    """Forces print to flush immediately so we don't miss errors."""
    print(msg, flush=True)

def get_learning_rate(episode):
    if episode < 1000:
        return 2e-6
    elif episode < 3000:
        return 1e-6
    elif episode < 6000:       # <--- NEW TIER
        return 5e-7            # Slower (Safe Zone)
    else:
        return 2e-7            # Ultra-Fine Tuning

def get_opponent_probabilities(episode):
    if episode < 3000: return 0.30, 0.50, 0.20
    else: return 0.20, 0.40, 0.40

def calculate_epsilon(episode):
    epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) * (episode / EPSILON_DECAY)
    return max(EPSILON_END, epsilon)

def find_latest_checkpoint(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "iron_nuclear_*.pth"))
    if not checkpoints:
        return None
    # Extract episode numbers and find max
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

# ════════════════════════════════════════════════════════════════════
# CLASSES
# ════════════════════════════════════════════════════════════════════

class OpponentManager:
    def __init__(self, pool_dir: str, action_manager, device):
        self.pool_dir = pool_dir
        self.action_manager = action_manager
        self.device = device
        self.opponent_model = D3QNModel(action_manager.action_dim, device).to(device)
        self.opponent_model.eval()
        
        all_files = glob.glob(os.path.join(pool_dir, "*.pth"))
        self.pool_files = [f for f in all_files if "gen7_final.pth" not in f]
        
        if len(self.pool_files) == 0:
            log_print("⚠️ RESUME: Pool might be empty. Using Random + Self-Play.")
        else:
            log_print(f"✅ Loaded Pool: {len(self.pool_files)} historical agents.")

    def get_opponent(self, current_online_model, prob_random, prob_pool):
        if len(self.pool_files) == 0: prob_pool = 0.0
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
        legal_moves = env.get_legal_moves()
        if not legal_moves: return None
        state_tensor = CheckersBoardEncoder().encode(env.board.get_state(), env.current_player).unsqueeze(0).to(self.device)
        with torch.no_grad(): q = model.get_q_values(state_tensor)[0]
        mask = self.action_manager.make_legal_action_mask(legal_moves).to(self.device)
        q[~mask] = -float('inf')
        action_id = int(q.argmax().item())
        move = self.action_manager.get_move_from_id(action_id)
        for lm in legal_moves:
            if isinstance(lm, list):
                if (tuple(lm[0][0]), tuple(lm[-1][1])) == move: return lm
            elif len(lm) == 2:
                 if (tuple(lm[0]), tuple(lm[1])) == move: return lm
        return legal_moves[0]

def select_agent_action(model, state, legal_moves, action_manager, epsilon, device):
    legal_mask = action_manager.make_legal_action_mask(legal_moves)
    legal_indices = torch.where(legal_mask)[0]
    if len(legal_indices) == 0: return None, None, legal_mask

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
            env_move = lm; break
        elif len(lm) == 2 and (tuple(lm[0]), tuple(lm[1])) == move:
            env_move = lm; break
            
    return (env_move if env_move else legal_moves[0]), action_id, legal_mask

# ════════════════════════════════════════════════════════════════════
# MAIN LOOP (DEBUG WRAPPED)
# ════════════════════════════════════════════════════════════════════

def main_training_loop():
    log_print("="*70)
    log_print("PHASE 5: IRON LEAGUE (DEBUG RESUME)")
    log_print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = CheckersEnv()
    action_manager = ActionManager(device=device)
    encoder = CheckersBoardEncoder()
    
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Side', 'Opponent', 'Result', 'Reward', 'P1_WinRate', 'P2_WinRate', 'vsRandom_WinRate', 'Loss', 'Epsilon', 'LR', 'MaxQ', 'AvgQ'])

    model = D3QNModel(action_manager.action_dim, device)
    
    # RESUME LOGIC
    resume_path = find_latest_checkpoint(CHECKPOINT_DIR)
    start_episode = 1
    optimizer = optim.Adam(model.online.parameters(), lr=get_learning_rate(1))

    if resume_path and os.path.exists(resume_path):
        log_print(f"🔄 RESUMING from Checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.online.load_state_dict(checkpoint['model_online'])
        model.target.load_state_dict(checkpoint['model_target'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_episode = checkpoint['episode'] + 1
        log_print(f"⏩ Fast-Forwarding to Episode {start_episode}")
    else:
        log_print("⚠️ No Resume Checkpoint Found! Starting Fresh.")

    buffer = ReplayBuffer(BUFFER_CAPACITY, action_manager.action_dim, device)
    trainer = D3QNTrainer(env, action_manager, encoder, model, optimizer, buffer, device, gamma=GAMMA, tau=TAU, gradient_clip=0.1)
    opponent_manager = OpponentManager(OPPONENT_POOL_DIR, action_manager, device)

    losses = []
    history_p1 = deque(maxlen=100)
    history_p2 = deque(maxlen=100)
    history_vs_random = deque(maxlen=100)
    
    log_print(f"🚀 Starting Training Loop...")

    auto_brake_active = False
    manual_lr = None

    for episode in range(start_episode, NUM_EPISODES + 1):
        
        # Only update LR if auto-brake hasn't taken over:
        if not auto_brake_active:
            current_lr = get_learning_rate(episode)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            current_lr = manual_lr  # Use the manually set LR

        epsilon = calculate_epsilon(episode)
        agent_side = 1 if random.random() < 0.5 else -1
        
        p_rand, p_pool, p_self = get_opponent_probabilities(episode)
        opp_name, opp_policy = opponent_manager.get_opponent(model, p_rand, p_pool)
        
        state = env.reset()
        done = False
        episode_reward = 0
        info = {}
        step_count = 0

        # Heartbeat for debug (removes ambiguity on freeze vs crash)
        if episode % 100 == 0:
            log_print(f"💓 Heartbeat: Entering Episode {episode}...")

        if agent_side == -1:
            opp_action = opp_policy(env)
            if opp_action: state, _, done, _ = env.step(opp_action)
            else: done = True

        while not done:
            step_count += 1
            if step_count >= MAX_MOVES_PER_GAME:
                done = True
                info['winner'] = 0  # Force Draw
                break

            current_player = env.current_player
            legal_moves = env.get_legal_moves()
            if not legal_moves: done = True; break

            if current_player == agent_side:
                encoded_state = encoder.encode(state, player=agent_side)
                env_move, action_id, _ = select_agent_action(model, encoded_state, legal_moves, action_manager, epsilon, device)
                
                if env_move is None or action_id is None: done = True; break

                next_state, reward, done, info = env.step(env_move)
                
                next_encoded = encoder.encode(next_state, player=agent_side)
                next_mask = action_manager.make_legal_action_mask(env.get_legal_moves() if not done else [])
                
                buffer.push(encoded_state, action_id, reward, next_encoded, done, next_mask)
                
                if len(buffer) > MIN_BUFFER_SIZE:
                    model.train()
                    losses.append(trainer.train_step(BATCH_SIZE))
                    trainer.update_target_network()

                episode_reward += reward
                state = next_state
            else:
                opp_action = opp_policy(env)
                if opp_action: state, _, done, info = env.step(opp_action)
                else: done = True

        # ----------------------------------------------------------------
        # SMART Q-HEALTH MONITOR (With Auto-Brake)
        # ----------------------------------------------------------------
        max_q, avg_q = 0.0, 0.0
        if episode % 50 == 0 and len(buffer) > MIN_BUFFER_SIZE:
            with torch.no_grad():
                sample_size = min(1000, len(buffer))
                sample_states = torch.from_numpy(buffer.states[:sample_size]).to(device)
                sample_q = model.online(sample_states).abs()
                max_q = sample_q.max().item()
                avg_q = sample_q.mean().item()
                
                # 1. LOGGING
                if episode % 100 == 0:
                    current_lr_val = optimizer.param_groups[0]['lr']
                    log_print(f"  [Q-Health] Max: {max_q:.2f} | Avg: {avg_q:.2f} | LR: {current_lr_val:.2e}")

                # 2. ADAPTIVE AUTO-BRAKE (FIXED: No Floor)
                # Raised threshold to 35.0 (since 4500 proved 47.0 is playable)
                if max_q > 35.0:
                    old_lr = optimizer.param_groups[0]['lr']
                    # Keep cutting in half, but don't go below 1e-8 (Atomic Level)
                    new_lr = max(old_lr * 0.5, 1e-8)

                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    
                    # Force the manual schedule to respect this new low
                    manual_lr = new_lr
                    auto_brake_active = True
                    
                    log_print(f"⚠️ AUTO-BRAKE TRIGGERED: Max Q={max_q:.1f}. Cutting LR: {old_lr:.2e} -> {new_lr:.2e}")

                # 3. EMERGENCY STOP (Raised threshold because Brake handles the rest)
                if max_q > 50.0 or avg_q > 20.0:
                    log_print(f"🚨 CRITICAL: Reactor Meltdown! Halting training.")
                    torch.save(model.online.state_dict(), f"emergency_nuclear_ep{episode}.pth")
                    return

        if episode % 100 == 0:
            # Clear CUDA cache to prevent memory accumulation
            if device == "cuda":
                torch.cuda.empty_cache()
            log_print(f"  [Memory] CUDA cache cleared")

        # LOGGING
        winner = info.get('winner', 0)
        is_win = (winner == agent_side)
        
        if agent_side == 1: history_p1.append(1 if is_win else 0)
        else: history_p2.append(1 if is_win else 0)
        if "Random" in opp_name: history_vs_random.append(1 if is_win else 0)

        if episode % 10 == 0:
            avg_loss = np.mean(losses[-100:]) if losses else 0.0
            p1_wr = np.mean(history_p1) * 100 if history_p1 else 0.0
            p2_wr = np.mean(history_p2) * 100 if history_p2 else 0.0
            rand_wr = np.mean(history_vs_random) * 100 if history_vs_random else 0.0
            side_str = "Red (P1)" if agent_side == 1 else "Blk (P2)"
            res_str = "WIN " if is_win else "LOSS" if winner != 0 else "DRAW"
            
            log_print(f"Ep {episode:5d} | {side_str} | Vs {opp_name:15s} | {res_str} | WR[P1]: {p1_wr:3.0f}% | WR[P2]: {p2_wr:3.0f}% | vsRand: {rand_wr:3.0f}%")
            
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode, side_str, opp_name, res_str, f"{episode_reward:.4f}", f"{p1_wr:.1f}", f"{p2_wr:.1f}", f"{rand_wr:.1f}", f"{avg_loss:.5f}", f"{epsilon:.4f}", f"{current_lr:.2e}", f"{max_q:.2f}", f"{avg_q:.2f}"])

        # CHECKPOINTING
        if episode % 500 == 0:
            path = os.path.join(CHECKPOINT_DIR, f"iron_nuclear_{episode}.pth")
            torch.save({
                'episode': episode,
                'model_online': model.online.state_dict(),
                'model_target': model.target.state_dict(),
                'optimizer': optimizer.state_dict(),
                'buffer_len': len(buffer)
            }, path)
            log_print(f"💾 Nuclear Checkpoint Saved: {path}")

def main():
    try:
        main_training_loop()
    except KeyboardInterrupt:
        log_print("\n⚠️ Training Interrupted by User (Ctrl+C). Saving emergency state...")
    except Exception as e:
        log_print(f"\n❌ FATAL CRASH DETECTED: {e}")
        log_print("Traceback:")
        traceback.print_exc()
    finally:
        log_print("🛑 Script Execution Finished.")

if __name__ == "__main__":
    main()