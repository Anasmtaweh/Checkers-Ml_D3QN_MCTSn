#!/usr/bin/env python3
"""
benchmark.py - D3QN Checkers Champion Finder (FIXED TRANSLATION)
"""

import torch
import os
import glob
import numpy as np
from checkers_env.env import CheckersEnv
from checkers_agents.random_agent import CheckersRandomAgent as RandomAgent
from training.common.action_manager import ActionManager
from training.common.board_encoder import CheckersBoardEncoder
from training.d3qn.model import D3QNModel

def find_matching_move(action_manager, action_id, legal_moves):
    """
    Translates the Network's Action ID into the Environment's specific Move Object.
    Crucial for handling multi-jumps correctly.
    """
    # 1. Get the simple move (Start, End) from the ID
    move = action_manager.get_move_from_id(action_id)
    
    # 2. Search legal_moves for the matching complex object
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
            # Simple move (Tuple of coordinates)
            if len(legal_move) == 2:
                if (tuple(legal_move[0]), tuple(legal_move[1])) == move:
                    env_move = legal_move
                    break
    
    # Fallback: If exact match fails, picking the first legal move is better than crashing,
    # but usually this logic finds the correct one.
    return env_move if env_move is not None else legal_moves[0]

def run_benchmark():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on {device}...\n")
    
    checkpoints = sorted(glob.glob("checkpoints/*.pth"))
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
        model = D3QNModel(action_dim=action_manager.action_dim, device=device).to(device)
        try:
            data = torch.load(cp, map_location=device)
            if "model_online" in data:
                model.online.load_state_dict(data["model_online"])
            elif "online_model_state_dict" in data:
                model.online.load_state_dict(data["online_model_state_dict"])
            else:
                model.online.load_state_dict(data)
            model.eval()
        except:
            continue

        wins_p1 = 0
        wins_p2 = 0
        total_games_per_side = 25
        
        # --- PLAY AS P1 (RED) ---
        for _ in range(total_games_per_side):
            state = env.reset()
            done = False
            steps = 0
            info = {}
            
            while not done and steps < 200:
                current = env.current_player
                legal = env.get_legal_moves()
                if not legal:
                    done = True
                    break
                
                if current == 1: # Agent
                    encoded_state = encoder.encode(state, player=1).unsqueeze(0).to(device)
                    
                    mask = action_manager.make_legal_action_mask(legal).to(device)
                    
                    # 5% Random noise to break loops (Matches Training)
                    if np.random.random() < 0.05:
                        legal_indices = torch.where(mask)[0]
                        action_id = int(legal_indices[np.random.randint(len(legal_indices))].item())
                    else:
                        with torch.no_grad():
                            q = model.get_q_values(encoded_state)[0]
                        q[~mask] = -float('inf')
                        action_id = int(q.argmax().item())
                    
                    # --- THE FIX: TRANSLATE ACTION ---
                    env_action = find_matching_move(action_manager, action_id, legal)
                    
                    state, _, done, info = env.step(env_action)
                else: # Random Opponent
                    action = opponent.select_action(env)
                    state, _, done, info = env.step(action)
                steps += 1
            
            if info.get('winner') == 1:
                wins_p1 += 1

        # --- PLAY AS P2 (BLACK) ---
        for _ in range(total_games_per_side):
            state = env.reset()
            done = False
            steps = 0
            info = {}
            
            while not done and steps < 200:
                current = env.current_player
                legal = env.get_legal_moves()
                if not legal:
                    done = True
                    break
                
                if current == 2: # Agent
                    encoded_state = encoder.encode(state, player=2).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q = model.get_q_values(encoded_state)[0]
                    mask = action_manager.make_legal_action_mask(legal).to(device)
                    q[~mask] = -float('inf')
                    action_id = int(q.argmax().item())
                    
                    # --- THE FIX: TRANSLATE ACTION ---
                    env_action = find_matching_move(action_manager, action_id, legal)
                    
                    state, _, done, info = env.step(env_action)
                else:
                    action = opponent.select_action(env)
                    state, _, done, info = env.step(action)
                steps += 1
            
            w = info.get('winner')
            if w == 2 or w == -1:
                wins_p2 += 1

        total = wins_p1 + wins_p2
        print(f"Model: {os.path.basename(cp):30} | P1: {wins_p1}/25 | P2: {wins_p2}/25 | Total: {total*2:.0f}%")
        results.append((cp, total*2))

    results.sort(key=lambda x: x[1], reverse=True)
    if results:
        print(f"\n🏆 CHAMPION: {os.path.basename(results[0][0])} ({results[0][1]}%)")

if __name__ == "__main__":
    run_benchmark()