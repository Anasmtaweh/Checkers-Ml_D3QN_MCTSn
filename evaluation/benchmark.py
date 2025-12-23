#!/usr/bin/env python3
"""
benchmark.py - D3QN Checkers Champion Finder (STRICT & FIXED)
"""

import torch
import os
import glob
import sys
import numpy as np
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.game import CheckersEnv
from core.action_manager import ActionManager
from core.board_encoder import CheckersBoardEncoder
from training.d3qn.model import D3QNModel


class RandomAgent:
    """Simple agent that plays random moves."""
    def select_action(self, env):
        legal_moves = env.get_legal_moves()
        return random.choice(legal_moves) if legal_moves else None

def find_matching_move(action_manager, action_id, legal_moves):
    """
    Translates the Network's Action ID into the Environment's specific Move Object.
    Crucial for handling multi-jumps correctly.
    """
    move = action_manager.get_move_from_id(action_id)
    
    env_move = None
    for legal_move in legal_moves:
        if isinstance(legal_move, list):
            # Capture sequence (List of steps)
            if legal_move:
                start = tuple(legal_move[0][0])
                landing = tuple(legal_move[-1][1])
                if (start, landing) == move:
                    env_move = legal_move
                    break
        else:
            # Simple move
            if len(legal_move) == 2:
                if (tuple(legal_move[0]), tuple(legal_move[1])) == move:
                    env_move = legal_move
                    break
    
    # Fallback to first legal move if mapping fails (safety net)
    return env_move if env_move is not None else legal_moves[0]

def run_benchmark():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on {device} (Deterministic Mode)...\n")
    
    # Scan multiple directories for models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    search_patterns = [
        os.path.join(project_root, "checkpoints", "*.pth"),
        os.path.join(script_dir, "checkpoints_iron_league_v3", "*.pth"),
        os.path.join(script_dir, "opponent_pool", "*.pth"),
        os.path.join(project_root, "opponent_pool", "*.pth")
    ]
    
    checkpoints = []
    for pattern in search_patterns:
        checkpoints.extend(glob.glob(pattern))
    
    checkpoints = sorted(list(set(checkpoints)))

    if not checkpoints:
        print("No checkpoints found!")
        return

    results = []

    # Init Components
    env = CheckersEnv()
    action_manager = ActionManager(device=device)
    encoder = CheckersBoardEncoder()
    opponent = RandomAgent()
    
    for cp in checkpoints:
        # Load Model
        model = D3QNModel(action_dim=action_manager.action_dim, device=device).to(device)
        try:
            data = torch.load(cp, map_location=device)
            # Handle different saving formats
            if isinstance(data, dict):
                if "model_online" in data:
                    model.online.load_state_dict(data["model_online"])
                elif "online_model_state_dict" in data:
                    model.online.load_state_dict(data["online_model_state_dict"])
                else:
                    # Fallback if state dict is at root
                    model.online.load_state_dict(data)
            else:
                model.online.load_state_dict(data)
            
            model.eval()
        except Exception as e:
            print(f"Skipping {cp}: {e}")
            continue

        wins_p1 = 0
        wins_p2 = 0
        total_games_per_side = 25
        max_steps = 200  # Prevent infinite loops in deterministic play
        
        # ---------------------------------------------------------
        # ROUND 1: AGENT IS PLAYER 1 (RED)
        # ---------------------------------------------------------
        for _ in range(total_games_per_side):
            state = env.reset()
            done = False
            steps = 0
            info = {}
            
            while not done and steps < max_steps:
                current = env.current_player
                legal = env.get_legal_moves()
                if not legal:
                    done = True # No moves = Loss
                    break
                
                if current == 1: # AGENT
                    # Encode for Player 1
                    encoded_state = encoder.encode(state, player=1).unsqueeze(0).to(device)
                    
                    # Get Greedy Action (No Randomness)
                    with torch.no_grad():
                        q = model.get_q_values(encoded_state)[0]
                    
                    mask = action_manager.make_legal_action_mask(legal).to(device)
                    q[~mask] = -float('inf') # Mask illegal moves
                    action_id = int(q.argmax().item()) # Strictly Greedy
                    
                    env_action = find_matching_move(action_manager, action_id, legal)
                    state, _, done, info = env.step(env_action)
                
                else: # OPPONENT (Player -1)
                    action = opponent.select_action(env)
                    state, _, done, info = env.step(action)
                
                steps += 1
            
            # Check Winner
            if info.get('winner') == 1:
                wins_p1 += 1

        # ---------------------------------------------------------
        # ROUND 2: AGENT IS PLAYER 2 (BLACK)
        # ---------------------------------------------------------
        for _ in range(total_games_per_side):
            state = env.reset()
            done = False
            steps = 0
            info = {}
            
            while not done and steps < max_steps:
                current = env.current_player
                legal = env.get_legal_moves()
                if not legal:
                    done = True
                    break
                
                if current == -1: # AGENT (Fixed ID from 2 to -1)
                    # Encode for Player -1 (Env rotates board for us, but we must pass correct ID)
                    encoded_state = encoder.encode(state, player=-1).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        q = model.get_q_values(encoded_state)[0]
                        
                    mask = action_manager.make_legal_action_mask(legal).to(device)
                    q[~mask] = -float('inf')
                    action_id = int(q.argmax().item())
                    
                    env_action = find_matching_move(action_manager, action_id, legal)
                    state, _, done, info = env.step(env_action)
                
                else: # OPPONENT (Player 1)
                    action = opponent.select_action(env)
                    state, _, done, info = env.step(action)
                
                steps += 1
            
            # Check Winner
            if info.get('winner') == -1:
                wins_p2 += 1

        total = wins_p1 + wins_p2
        print(f"Model: {os.path.basename(cp):30} | P1: {wins_p1}/25 | P2: {wins_p2}/25 | Total: {total*2:.0f}%")
        results.append((cp, total*2))

    # Sort and Display Best
    results.sort(key=lambda x: x[1], reverse=True)
    if results:
        print(f"\n🏆 CHAMPION: {os.path.basename(results[0][0])} ({results[0][1]}%)")

if __name__ == "__main__":
    run_benchmark()