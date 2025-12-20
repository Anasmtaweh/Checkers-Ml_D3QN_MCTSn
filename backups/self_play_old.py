#!/usr/bin/env python3
"""
self_play.py - Phase 5: Iron League (STABLE RECOVERY)

Updates:
1. Learning Rate reduced to 2e-6 (Success Paradox Fix).
2. Adaptive Learning Rate Reduction added (Auto-brakes).
3. Resumes from Episode 1000 (The Peak).

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
LEARNING_RATE = 2e-6  # <--- FIXED: Lower LR for advanced agent stability
TAU = 0.001

EPSILON_START = 0.30
EPSILON_END = 0.05
EPSILON_DECAY = 5000

BUFFER_CAPACITY = 40000  
MIN_BUFFER_SIZE = 5000   

REWARD_SCALE = 1.0
OPPONENT_POOL_DIR = "opponent_pool"
CHECKPOINT_DIR = "checkpoints_iron_league"
LOG_FILE = "iron_league_log_v2.csv" # New log file

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
MIN_POOL_SIZE = 3 

# ════════════════════════════════════════════════════════════════════
# DYNAMIC PROBABILITIES
# ════════════════════════════════════════════════════════════════════

def get_opponent_probabilities(episode):
    if episode < 3000:
        return 0.30, 0.50, 0.20
    else:
        return 0.20, 0.40, 0.40

# ════════════════════════════════════════════════════════════════════
# OPPONENT MANAGER
# ════════════════════════════════════════════════════════════════════

class OpponentManager:
    def __init__(self, pool_dir: str, action_manager, device):
        self.pool_dir = pool_dir
        self.action_manager = action_manager
        self.device = device
        self.opponent_model = D3QNModel(action_manager.action_dim, device).to(device)
        self.opponent_model.eval()
        
        self.pool_files = glob.glob(os.path.join(pool_dir, "*.pth"))
        
        if len(self.pool_files) < MIN_POOL_SIZE:
            print(f"⚠️ Warning: Pool small ({len(self.pool_files)}). Ensure these are high quality!")
        
        print(f"✅ Loaded Pool: {len(self.pool_files)} historical agents.")

    def get_opponent(self, current_online_model, prob_random, prob_pool):
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

# ════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════

def calculate_epsilon(episode):
    epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) * (episode / EPSILON_DECAY)
    return max(EPSILON_END, epsilon)

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
# MAIN LOOP
# ════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("PHASE 5: IRON LEAGUE (RECOVERY MODE)")
    print(f"Learning Rate: {LEARNING_RATE} | Adaptive Reduction: ON")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = CheckersEnv()
    action_manager = ActionManager(device=device)
    encoder = CheckersBoardEncoder()
    
    # Initialize CSV Log
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Side', 'Opponent', 'Result', 'Reward', 'P1_WinRate', 'P2_WinRate', 'vsRandom_WinRate', 'Loss', 'Epsilon', 'LR'])

    model = D3QNModel(action_manager.action_dim, device)
    
    # ----------------------------------------------------------------
    # LOAD FROM GEN 7 FINAL (ROBUST LOADING FIX)
    # ----------------------------------------------------------------
    start_model_path = "opponent_pool/gen7_final.pth"
    if os.path.exists(start_model_path):
        print(f"✅ Loading Base Model: {start_model_path}")
        checkpoint = torch.load(start_model_path, map_location=device)
        
        # FIX: Check if it's a full checkpoint dict or just weights
        if isinstance(checkpoint, dict) and 'model_online' in checkpoint:
            # It's a full checkpoint (has keys 'model_online', 'optimizer', etc.)
            model.online.load_state_dict(checkpoint['model_online'])
            if 'model_target' in checkpoint:
                model.target.load_state_dict(checkpoint['model_target'])
            else:
                model.target.load_state_dict(checkpoint['model_online'])
        else:
            # It's just the raw state_dict (Old format)
            print("   (Detected raw state_dict format)")
            model.online.load_state_dict(checkpoint)
            model.target.load_state_dict(checkpoint)
            
    else:
        raise FileNotFoundError(f"Missing {start_model_path}. Please set up the pool.")

    optimizer = optim.Adam(model.online.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer(BUFFER_CAPACITY, action_manager.action_dim, device)
    
    trainer = D3QNTrainer(env, action_manager, encoder, model, optimizer, buffer, device, gamma=GAMMA, tau=TAU, gradient_clip=0.1)
    opponent_manager = OpponentManager(OPPONENT_POOL_DIR, action_manager, device)

    losses = []
    history_p1 = deque(maxlen=100)
    history_p2 = deque(maxlen=100)
    history_vs_random = deque(maxlen=100)
    
    # Resume episode count if needed, but we'll start loop from 1
    # You can mentally add 1000 to the episode count or restart the count
    
    print(f"\nTraining for {NUM_EPISODES} episodes...")

    for episode in range(1, NUM_EPISODES + 1):
        epsilon = calculate_epsilon(episode)
        agent_side = 1 if random.random() < 0.5 else -1
        
        p_rand, p_pool, p_self = get_opponent_probabilities(episode)
        opp_name, opp_policy = opponent_manager.get_opponent(model, p_rand, p_pool)
        
        state = env.reset()
        done = False
        episode_reward = 0
        info = {}

        if episode == 1 and agent_side == -1:
            enc_p1 = encoder.encode(state, 1)
            enc_p2 = encoder.encode(state, -1)
            if torch.equal(enc_p1, enc_p2):
                print("🚨 CRITICAL ERROR: Board Encoder is not rotating for Player 2!")
                return

        if agent_side == -1:
            opp_action = opp_policy(env)
            if opp_action: state, _, done, _ = env.step(opp_action)
            else: done = True

        while not done:
            current_player = env.current_player
            legal_moves = env.get_legal_moves()
            if not legal_moves: done = True; break

            if current_player == agent_side:
                encoded_state = encoder.encode(state, player=agent_side)
                env_move, action_id, _ = select_agent_action(model, encoded_state, legal_moves, action_manager, epsilon, device)
                
                if env_move is None or action_id is None: done = True; break

                next_state, reward, done, info = env.step(env_move)
                
                winner = info.get('winner', 0)
                if done and winner == agent_side: custom_reward = 1.0
                elif done and winner == -agent_side: custom_reward = -1.0
                elif done: custom_reward = 0.0
                else:
                    if reward > 20.0: custom_reward = 0.01
                    elif reward > 8.0: custom_reward = 0.001
                    else: custom_reward = -0.0001
                
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
                opp_action = opp_policy(env)
                if opp_action: state, _, done, info = env.step(opp_action)
                else: done = True

        # ════════════════════════════════════════════════════════════════════
        # ADAPTIVE SAFETY SYSTEM (Fix #3)
        # ════════════════════════════════════════════════════════════════════
        if episode % 50 == 0 and len(buffer) > MIN_BUFFER_SIZE:
            with torch.no_grad():
                sample_size = min(1000, len(buffer))
                sample_states = torch.from_numpy(buffer.states[:sample_size]).to(device)
                
                sample_q_values = model.online(sample_states).abs()
                max_q = sample_q_values.max().item()
                avg_q = sample_q_values.mean().item()
                
                if episode % 100 == 0:
                     print(f"  [Q-Health] Max: {max_q:.2f} | Avg: {avg_q:.2f}")

                # 1. EMERGENCY STOP (Catastrophe Prevention)
                if max_q > 50.0 or avg_q > 10.0:
                    print(f"🚨 CRITICAL: Q-Values Exploding!")
                    print(f"   Max Q: {max_q:.2f} | Avg Q: {avg_q:.2f}")
                    torch.save({
                        'episode': episode,
                        'model_online': model.online.state_dict(),
                        'model_target': model.target.state_dict()
                    }, f"emergency_stop_ep{episode}.pth")
                    print("   TRAINING HALTED.")
                    return

                # 2. ADAPTIVE LEARNING RATE (The "Auto-Brake")
                if max_q > 30.0 and optimizer.param_groups[0]['lr'] > 1e-6:
                    old_lr = optimizer.param_groups[0]['lr']
                    new_lr = old_lr * 0.5
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"⚠️ AUTO-ADJUSTING: Q={max_q:.1f} too high, reducing LR: {old_lr:.2e} → {new_lr:.2e}")

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
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Ep {episode:5d} | {side_str} | Vs {opp_name:15s} | {res_str} | WR[P1]: {p1_wr:3.0f}% | WR[P2]: {p2_wr:3.0f}% | vsRand: {rand_wr:3.0f}%")
            
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode, side_str, opp_name, res_str, f"{episode_reward:.4f}", f"{p1_wr:.1f}", f"{p2_wr:.1f}", f"{rand_wr:.1f}", f"{avg_loss:.5f}", f"{epsilon:.4f}", f"{current_lr:.2e}"])

        if episode % 1000 == 0:
            path = os.path.join(CHECKPOINT_DIR, f"iron_agent_v2_{episode}.pth")
            torch.save(model.online.state_dict(), path)
            print(f"💾 Saved Checkpoint: {path}")

if __name__ == "__main__":
    main()