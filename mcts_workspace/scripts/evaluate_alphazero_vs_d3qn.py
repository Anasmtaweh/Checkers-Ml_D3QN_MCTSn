import os
import sys
import glob
import re
import pandas as pd
import matplotlib
# Force headless backend for server usage
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm

# Add workspaces to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../d3qn_workspace')))

# Imports from MCTS workspace core
from mcts_workspace.core.game import CheckersEnv
from mcts_workspace.core.action_manager import ActionManager
from mcts_workspace.core.board_encoder import CheckersBoardEncoder
from mcts_workspace.core.move_parser import parse_legal_moves

import torch.nn as nn
import torch.nn.functional as F

# Import AlphaZero classes
from mcts_workspace.training.alpha_zero.network import AlphaZeroModel
from mcts_workspace.training.alpha_zero.mcts import MCTS

# ==================================================================================
# CONFIGURATION
# ==================================================================================
MCTS_SIMULATIONS = 60    # 60 is fast and sufficient to beat a static D3QN agent
MCTS_TEMP = 0.0          # Deterministic play
D3QN_EPSILON = 0.0       # Deterministic play
MAX_MOVES_PER_GAME = 150 # Reduced to 150 as requested

# ==================================================================================
# D3QN MODEL DEFINITION (Local copy to support 5 channels)
# ==================================================================================

class DuelingDQN(nn.Module):
    def __init__(self, action_dim: int, device: str = "cpu", in_channels: int = 5):
        super(DuelingDQN, self).__init__()
        self.device = torch.device(device)
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.flatten_size = 64 * 8 * 8
        self.fc_norm = nn.LayerNorm(self.flatten_size)
        
        self.value_fc1 = nn.Linear(self.flatten_size, 512)
        self.value_fc2 = nn.Linear(512, 1)
        
        self.advantage_fc1 = nn.Linear(self.flatten_size, 512)
        self.advantage_fc2 = nn.Linear(512, action_dim)

        self.to(self.device)

    def forward(self, x: torch.Tensor, player_side: int = 1) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        # SLICING FIX: Handle 6-channel input for 5-channel model
        if x.shape[1] > self.in_channels:
             x = x[:, :self.in_channels, :, :]
        
        x = x.to(self.device)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = self.fc_norm(x)
        
        val = F.relu(self.value_fc1(x))
        val = self.value_fc2(val)
        
        adv = F.relu(self.advantage_fc1(x))
        adv = self.advantage_fc2(adv)

        return val + (adv - adv.mean(dim=1, keepdim=True))

    def get_q_values(self, state: torch.Tensor, player_side: int = 1) -> torch.Tensor:
        return self.forward(state, player_side)

class D3QNModel:
    def __init__(self, action_dim: int, device: str = "cpu", in_channels: int = 5):
        self.online = DuelingDQN(action_dim, device, in_channels)
        self.target = DuelingDQN(action_dim, device, in_channels)
        self.target.eval()

    def get_q_values(self, state, player_side=1, use_target=False):
        net = self.target if use_target else self.online
        return net.get_q_values(state, player_side)
    
    def eval(self):
        self.online.eval()
        self.target.eval()

# ==================================================================================
# ADAPTED D3QN AGENT
# ==================================================================================

class D3QNAgent:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.encoder = CheckersBoardEncoder()
        self.action_manager = ActionManager(device=self.device)
        self.model = D3QNModel(
            action_dim=self.action_manager.action_dim,
            device=str(self.device),
            in_channels=5
        )

    def load_weights(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "online" in state and "target" in state:
            self.model.online.load_state_dict(state["online"])
            self.model.target.load_state_dict(state["target"])
        elif isinstance(state, dict) and "model" in state:
            model_state = state["model"]
            if "online" in model_state:
                self.model.online.load_state_dict(model_state["online"])
                self.model.target.load_state_dict(model_state["target"])
            else:
                self.model.online.load_state_dict(model_state)
        elif isinstance(state, dict) and "model_online" in state:
            self.model.online.load_state_dict(state["model_online"])
            self.model.target.load_state_dict(state["model_target"])
        else:
            self.model.online.load_state_dict(state)
        
        self.model.online.to(self.device)
        self.model.target.to(self.device)

    def select_action(self, board, player, legal_moves, epsilon: float = 0.0, info: Optional[Dict] = None) -> Tuple[Optional[Any], Optional[int]]:
        normalized_moves, mapping = parse_legal_moves(legal_moves, self.action_manager)
        if not mapping: return None, None

        if player == -1:
            canonical_moves = [self.action_manager.flip_move(m) for m in normalized_moves]
            mask = self.action_manager.make_legal_action_mask(canonical_moves)
            canonical_to_absolute = {}
            for i, cm in enumerate(canonical_moves):
                cid = self.action_manager.get_action_id(cm)
                if cid >= 0:
                    orig_move = normalized_moves[i]
                    aid = self.action_manager.get_action_id(orig_move)
                    canonical_to_absolute[cid] = aid
        else:
            mask = self.action_manager.make_legal_action_mask(normalized_moves)
            canonical_to_absolute = None

        action_index = None
        self.model.eval()
        with torch.no_grad():
            if epsilon > 0 and random.random() < epsilon:
                absolute_id = random.choice(list(mapping.keys()))
                return mapping[absolute_id], absolute_id
            else:
                if info is None: info = {}
                state = self.encoder.encode(board, player=player, force_move_from=info.get('force_capture_from'))
                if state.dim() == 3: state = state.unsqueeze(0)
                state = state.to(self.device)

                q_values = self.model.get_q_values(state)

                if torch.isnan(q_values).any() or torch.isinf(q_values).any():
                    absolute_id = random.choice(list(mapping.keys()))
                    return mapping[absolute_id], absolute_id
                else:
                    masked = q_values.clone()
                    masked[mask.unsqueeze(0) == 0] = -1e9
                    action_index = int(torch.argmax(masked, dim=1).item())

        absolute_id = None
        if action_index is not None:
            if player == -1 and canonical_to_absolute is not None:
                absolute_id = canonical_to_absolute.get(action_index)
            else:
                absolute_id = action_index

        if absolute_id is not None and absolute_id in mapping:
            return mapping[absolute_id], absolute_id
            
        absolute_id = random.choice(list(mapping.keys()))
        return mapping[absolute_id], absolute_id

# ==================================================================================
# EVALUATION LOGIC
# ==================================================================================

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def play_match(env, az_agent, d3qn_agent, action_manager, az_player_color, num_games=10):
    az_wins = 0
    d3qn_wins = 0
    draws = 0
    
    encoder = CheckersBoardEncoder()

    for _ in range(num_games):
        env.reset()
        done = False
        moves_count = 0
        
        # Anti-Shuffling: 3-fold repetition detection
        board_history = defaultdict(int)
        
        while not done:
            # 1. Check Move Limit
            moves_count += 1
            if moves_count > MAX_MOVES_PER_GAME:
                done = True
                env.winner = 0 # Force Draw
                break
            
            # 2. Check Repetition (Speed Optimization)
            # Create a hashable signature of the board (bytes)
            board_sig = env.board.board.tobytes()
            board_history[board_sig] += 1
            if board_history[board_sig] >= 3:
                # 3-fold repetition detected -> Immediate Draw
                done = True
                env.winner = 0
                break

            # 3. Standard Game Logic
            player = env.current_player
            legal_moves = env.get_legal_moves()
            
            if not legal_moves:
                break 
                
            if player == az_player_color:
                # AlphaZero's turn
                mcts = MCTS(az_agent.network, action_manager, encoder, 
                            num_simulations=MCTS_SIMULATIONS, 
                            device=az_agent.device,
                            dirichlet_alpha=0.0)
                
                action_probs, _ = mcts.get_action_prob(env, temp=MCTS_TEMP, training=False)
                best_action_id = int(np.argmax(action_probs))
                move = mcts._get_move_from_action(best_action_id, legal_moves, player)
                if move is None: move = random.choice(legal_moves)
            else:
                # D3QN's turn
                info = {'force_capture_from': env.force_capture_from}
                move, _ = d3qn_agent.select_action(env.board.board, player, legal_moves, epsilon=D3QN_EPSILON, info=info)
            
            _, _, done, _ = env.step(move)
            
        if env.winner == 0:
            draws += 1
        elif env.winner == az_player_color:
            az_wins += 1
        else:
            d3qn_wins += 1

    return az_wins, d3qn_wins, draws

def evaluate_checkpoints(checkpoints_dir, d3qn_path, output_dir, games_per_side=10):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    action_manager = ActionManager(device=device)
    
    print(f"Loading D3QN Agent from {d3qn_path}...")
    d3qn_agent = D3QNAgent(device=device)
    try:
        d3qn_agent.load_weights(d3qn_path)
    except Exception as e:
        print(f"Error loading D3QN weights: {e}")
        return

    checkpoints = glob.glob(os.path.join(checkpoints_dir, "checkpoint_iter_*.pth"))
    checkpoints.sort(key=natural_sort_key)
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoints_dir}")
        return

    print(f"Found {len(checkpoints)} AlphaZero checkpoints.")
    print(f"Simulations per move: {MCTS_SIMULATIONS}")
    print(f"Max Moves per game: {MAX_MOVES_PER_GAME}")
    
    results = []
    az_model = AlphaZeroModel(action_dim=170, device=device)
    env = CheckersEnv()

    for ckpt_path in tqdm(checkpoints, desc="Evaluating Checkpoints"):
        match = re.search(r'iter_(\d+)', ckpt_path)
        if not match: continue
        iter_num = int(match.group(1))
        
        try:
            az_model.load(ckpt_path)
        except Exception as e:
            print(f"Failed to load {ckpt_path}: {e}")
            continue
            
        # 1. AZ (Red) vs D3QN (Black)
        az_wins_r, d3qn_wins_b, draws_r = play_match(env, az_model, d3qn_agent, action_manager, az_player_color=1, num_games=games_per_side)
        
        # 2. D3QN (Red) vs AZ (Black)
        az_wins_b, d3qn_wins_r, draws_b = play_match(env, az_model, d3qn_agent, action_manager, az_player_color=-1, num_games=games_per_side)
        
        total_games = 2 * games_per_side
        total_az_wins = az_wins_r + az_wins_b
        total_d3qn_wins = d3qn_wins_b + d3qn_wins_r
        total_draws = draws_r + draws_b
        
        win_rate = total_az_wins / total_games
        draw_rate = total_draws / total_games
        loss_rate = total_d3qn_wins / total_games
        
        results.append({
            "iteration": iter_num,
            "az_wins": total_az_wins,
            "d3qn_wins": total_d3qn_wins,
            "draws": total_draws,
            "win_rate": win_rate,
            "draw_rate": draw_rate,
            "loss_rate": loss_rate,
            "az_p1_wins": az_wins_r,
            "az_p2_wins": az_wins_b
        })
        
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)

    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values("iteration", inplace=True)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df["iteration"], df["win_rate"], label="AZ Win Rate", marker='o', color='green')
        plt.plot(df["iteration"], df["draw_rate"], label="Draw Rate", marker='x', linestyle='--', color='gray')
        plt.plot(df["iteration"], df["loss_rate"], label="D3QN Win Rate", marker='s', linestyle='-.', color='red')
        
        plt.title(f"AlphaZero vs D3QN (Sims={MCTS_SIMULATIONS}, MaxMoves={MAX_MOVES_PER_GAME})")
        plt.xlabel("AlphaZero Iteration")
        plt.ylabel("Rate")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "performance_plot.png"))
        print(f"Evaluation complete. Results saved to {output_dir}")
    else:
        print("No results to plot.")

if __name__ == "__main__":
    checkpoints_dir = "/home/anas/ML_Gen2/mcts_workspace/checkpoints/alphazero"
    d3qn_path = "/home/anas/ML_Gen2/agents/d3qn/gen7_specialist.pth"
    output_dir = "/home/anas/ML_Gen2/evaluation_results/alphazero_vs_d3qn"
    
    evaluate_checkpoints(checkpoints_dir, d3qn_path, output_dir, games_per_side=10)