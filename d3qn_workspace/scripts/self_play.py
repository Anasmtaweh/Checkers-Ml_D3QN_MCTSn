import torch
import torch.optim as optim
import numpy as np
import os
import sys
import random
import csv
from collections import deque

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..')))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

from core.game import CheckersEnv
from core.action_manager import ActionManager
from core.board_encoder import CheckersBoardEncoder
from training.d3qn.buffer import ReplayBuffer
from training.d3qn.model import D3QNModel
from training.d3qn.trainer import D3QNTrainer

# CONFIGURATION
NUM_EPISODES = 20000
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005 # Faster soft update
EPSILON_START = 0.50
EPSILON_END = 0.05
EPSILON_DECAY = 5000
BUFFER_CAPACITY = 50000
MIN_BUFFER_SIZE = 2000

CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "..", "checkpoints", "d3qn")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
LOG_FILE = os.path.join(SCRIPT_DIR, "..", "data", "d3qn_training_log.csv")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def determine_game_phase(board_state):
    """Returns 'endgame' if pieces <= 6, else 'normal'"""
    return "endgame" if np.count_nonzero(board_state) <= 6 else "normal"

def calculate_epsilon(episode):
    epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) * (episode / EPSILON_DECAY)
    return max(EPSILON_END, epsilon)

def select_agent_action(model, state, legal_moves, action_manager, epsilon, device):
    legal_mask = action_manager.make_legal_action_mask(legal_moves)
    if np.random.random() < epsilon:
        legal_indices = torch.where(legal_mask)[0]
        action_id = int(legal_indices[np.random.randint(len(legal_indices))].item())
    else:
        model.eval()
        with torch.no_grad():
            q = model.get_q_values(state.unsqueeze(0).to(device))[0]
            q[~legal_mask] = -1e9
            action_id = int(q.argmax().item())

    move = action_manager.get_move_from_id(action_id)
    # Map back to env move format
    env_move = None
    for lm in legal_moves:
        # Check simple moves
        if len(lm) == 2 and (tuple(lm[0]), tuple(lm[1])) == move:
            env_move = lm; break
        # Check capture chains (start, end)
        if isinstance(lm, list) and (tuple(lm[0][0]), tuple(lm[-1][1])) == move:
            env_move = lm; break
    
    return (env_move if env_move else legal_moves[0]), action_id, legal_mask

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    
    env = CheckersEnv()
    action_manager = ActionManager(device=device)
    encoder = CheckersBoardEncoder()
    
    # 6 Channels input
    model = D3QNModel(action_manager.action_dim, device=device).to(device)
    optimizer = optim.Adam(model.online.parameters(), lr=1e-4)
    buffer = ReplayBuffer(BUFFER_CAPACITY, action_manager.action_dim, device)
    trainer = D3QNTrainer(env, action_manager, encoder, model, optimizer, buffer, device)
    
    losses = []

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        epsilon = calculate_epsilon(episode)
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            step_count += 1
            legal_moves = env.get_legal_moves()
            if not legal_moves: break
                
            # 1. Phase Logic
            phase = determine_game_phase(env.board.get_state())
            
            # 2. Dynamic Epsilon for Endgame (Anti-Stagnation)
            current_epsilon = max(0.10, epsilon * 2) if phase == "endgame" else epsilon

            encoded_state = encoder.encode(state, player=env.current_player, force_move_from=env.force_capture_from)
            
            # Select Action
            env_move, action_id, _ = select_agent_action(
                model, encoded_state, legal_moves, action_manager, current_epsilon, device
            )

            next_state, reward, done, info = env.step(env_move)
            
            # 3. Dynamic Rewards
            custom_reward = reward
            if done:
                winner = info.get('winner', 0)
                if winner == env.current_player: custom_reward = 1.0
                elif winner == -env.current_player: custom_reward = -1.0
                else: custom_reward = -0.5 # Punish Draws
            else:
                # Living Tax
                custom_reward = -0.01 if phase == "endgame" else -0.001
                # Capture Bonus
                if reward > 0.01: custom_reward += 0.2
                elif reward > 0.001: custom_reward += 0.05
            
            next_force_from = info.get("from", None)
            next_encoded = encoder.encode(next_state, player=env.current_player, force_move_from=next_force_from)
            next_mask = action_manager.make_legal_action_mask(env.get_legal_moves() if not done else [])
            
            buffer.push(encoded_state, action_id, custom_reward, next_encoded, done, next_mask)
            
            if len(buffer) > MIN_BUFFER_SIZE:
                model.train()
                loss = trainer.train_step(BATCH_SIZE, player_side=env.current_player)
                losses.append(loss)
                trainer.update_target_network()
            
            state = next_state
            episode_reward += custom_reward
            
            if step_count > 300: break # Safety break

        if episode % 10 == 0:
            print(f"Episode {episode} | Reward: {episode_reward:.3f} | Epsilon: {epsilon:.3f}")
            
        if episode % 500 == 0:
             torch.save(model.online.state_dict(), f"{CHECKPOINT_DIR}/d3qn_{episode}.pth")

if __name__ == "__main__":
    main()