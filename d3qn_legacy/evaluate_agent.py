#!/usr/bin/env python3
"""
evaluate_agent.py - Evaluation script for D3QN agent.
"""
import torch
import time
import argparse
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from checkers_env.env import CheckersEnv
from checkers_agents.random_agent import CheckersRandomAgent as RandomAgent
from common.action_manager import ActionManager
from common.board_encoder import CheckersBoardEncoder
from d3qn_legacy.d3qn.model import D3QNModel

def match_move(action_manager, action_id, legal_moves):
    """
    Translates the Network's Action ID back to the exact object 
    the Environment expects (handling jumps/multi-jumps).
    """
    move_struct = action_manager.get_move_from_id(int(action_id))
    
    for lm in legal_moves:
        # Case 1: Multi-Jump or List format
        if isinstance(lm, list):
             if (tuple(lm[0][0]), tuple(lm[-1][1])) == move_struct: return lm
        # Case 2: Simple Tuple format
        elif len(lm) == 2:
             if (tuple(lm[0]), tuple(lm[1])) == move_struct: return lm
             
    # Fallback: If for some reason matching fails, play the first legal move
    return legal_moves[0]

def evaluate(checkpoint_path, agent_player=1, delay=0.5, num_games=1):
    # Auto-locate model if not found in root
    if not os.path.exists(checkpoint_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        search_dirs = [
            "checkpoints_gen11_decisive",
            "checkpoints_iron_league_v3", "checkpoints", "opponent_pool",
            os.path.join(script_dir, "checkpoints_gen11_decisive"),
            os.path.join(script_dir, "checkpoints_iron_league_v3"),
            os.path.join(script_dir, "checkpoints"),
            os.path.join(script_dir, "opponent_pool")
        ]
        for d in search_dirs:
            p = os.path.join(d, checkpoint_path)
            if os.path.exists(p):
                print(f"🔍 Found model in '{d}': {p}")
                checkpoint_path = p
                break

    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return

    env = CheckersEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {checkpoint_path}")
    print(f"Agent playing as: Player {agent_player} ({'Red' if agent_player==1 else 'Black'})")
    if num_games > 1: print(f"Simulating {num_games} games...")

    # Load Components
    action_manager = ActionManager(device=device)
    encoder = CheckersBoardEncoder()
    model = D3QNModel(action_dim=action_manager.action_dim, device=device).to(device)
    
    # Load Weights
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict):
            if "online_model_state_dict" in checkpoint:
                model.online.load_state_dict(checkpoint["online_model_state_dict"])
            elif "model_online" in checkpoint:
                model.online.load_state_dict(checkpoint["model_online"])
            else:
                model.online.load_state_dict(checkpoint)
        else:
            model.online.load_state_dict(checkpoint)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    model.eval()

    opponent = RandomAgent()
    results = {"win": 0, "loss": 0, "draw": 0}
    
    for i in range(num_games):
        state = env.reset()
        done = False
        info = {}
        verbose = (num_games == 1) 
        
        while not done:
            if verbose: time.sleep(delay)
            current_player = env.current_player
            legal_moves = env.get_legal_moves()
            
            if not legal_moves:
                done = True
                continue

            if current_player == agent_player:
                state_tensor = encoder.encode(state, player=current_player).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    q_values = model.online(state_tensor)
                
                legal_mask = action_manager.make_legal_action_mask(legal_moves).to(device).unsqueeze(0)
                masked_q = q_values.clone()
                masked_q[~legal_mask] = -float('inf')
                
                best_action_idx = torch.argmax(masked_q, dim=1).item()
                env_action = match_move(action_manager, best_action_idx, legal_moves)
                
                state, reward, done, info = env.step(env_action)
                
            else:
                action = opponent.select_action(env)
                state, reward, done, info = env.step(action)

        winner = info.get('winner', 0)
        if winner == agent_player:
            results["win"] += 1
        elif winner == 0:
            results["draw"] += 1
        else:
            results["loss"] += 1
            
    if num_games > 1:
        print(f"\n📊 RESULTS ({num_games} Games):")
        print(f"Wins:   {results['win']} ({results['win']/num_games*100:.1f}%)")
        print(f"Losses: {results['loss']} ({results['loss']/num_games*100:.1f}%)")
        print(f"Draws:  {results['draw']} ({results['draw']/num_games*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to checkpoint")
    parser.add_argument("--model", type=str, help="Alias for --path")
    parser.add_argument("--checkpoint", type=str, help="Alias for --path")
    parser.add_argument("--player", type=int, default=1, help="Play as 1 (Red) or 2 (Black)")
    parser.add_argument("--speed", type=float, default=0.2, help="Speed")
    parser.add_argument("--games", type=int, default=1, help="Number of games")
    
    args = parser.parse_args()
    
    model_path = args.path or args.model or args.checkpoint
    if not model_path:
        parser.error("You must provide --path, --model, or --checkpoint")
    
    evaluate(model_path, args.player, args.speed, args.games)
