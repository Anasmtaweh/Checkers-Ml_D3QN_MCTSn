#!/usr/bin/env python3
"""
tournament_alphazero.py - AlphaZero Civil War

Runs a Round Robin tournament between different checkpoints of your AlphaZero agent.
Used to verify progress and check if "Chaos Mode" broke the draw streak.
"""

import os
import sys
import glob
import itertools
import copy
import random
import torch
import numpy as np

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..')))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

from mcts_workspace.core.game import CheckersEnv
from mcts_workspace.core.action_manager import ActionManager
from mcts_workspace.core.board_encoder import CheckersBoardEncoder
from mcts_workspace.training.alpha_zero.network import AlphaZeroModel
from mcts_workspace.training.alpha_zero.mcts import MCTS

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "alphazero")

# Filter: Only test specific iterations to save time
# Example: [10, 20, 25, 28, 29, 30]
# If empty [], it scans ALL .pth files (can be slow)
TARGET_ITERS = [ 20, 25, 29, ] 

GAMES_PER_MATCH = 4      # 2 Red, 2 Black
MAX_MOVES = 150          # Match the training limit
MCTS_SIMS = 400          # Fast sims for tournament (400 is too slow for batch testing)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ════════════════════════════════════════════════════════════════════
# PLAYER CLASS
# ════════════════════════════════════════════════════════════════════

class Player:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.score = 0.0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.matches_played = 0

    def update_stats(self, score, w, l, d):
        self.score += score
        self.wins += w
        self.losses += l
        self.draws += d
        self.matches_played += 1

def load_player(path, manager):
    try:
        checkpoint = torch.load(path, map_location=DEVICE)
        model = AlphaZeroModel(action_dim=manager.action_dim, device=DEVICE)
        model.network.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        name = os.path.basename(path).replace("checkpoint_iter_", "Iter ").replace(".pth", "")
        return Player(name, model)
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")
        return None

# ════════════════════════════════════════════════════════════════════
# GAME ENGINE
# ════════════════════════════════════════════════════════════════════

def get_action(player, env, manager, encoder):
    # Tournament MCTS = Low Temp (Play to win)
    mcts = MCTS(
        player.model, manager, encoder,
        num_simulations=MCTS_SIMS,
        device=DEVICE,
        dirichlet_alpha=0.0, # No noise in tournament!
        draw_value=0.0       # Pure evaluation
    )
    
    # We use a temp of 0.1 to allow slight variety, or 0.0 for pureargmax
    probs, _ = mcts.get_action_prob(env, temp=0.1, training=False)
    action_id = int(np.argmax(probs))
    
    legal_moves = env.get_legal_moves()
    move = mcts._get_move_from_action(action_id, legal_moves, env.current_player)
    
    # Fallback (should never happen with MCTS)
    return move if move else legal_moves[0]

def play_game(p1, p2, env, manager, encoder):
    env.reset()
    moves = 0
    
    while not env.done:
        moves += 1
        if moves > MAX_MOVES:
            return 0 # Draw by timeout

        current_player = p1 if env.current_player == 1 else p2
        move = get_action(current_player, env, manager, encoder)
        
        _, _, done, info = env.step(move)
        
        if done:
            return info['winner']
            
    return env.winner

def run_match(p1, p2, env, manager, encoder):
    header = f"⚔️  {p1.name} vs {p2.name}"
    print(f"{header:<30} | Playing...", end="", flush=True)

    s1, w1, l1, d1 = 0, 0, 0, 0
    s2, w2, l2, d2 = 0, 0, 0, 0

    # P1 is Red
    for i in range(GAMES_PER_MATCH // 2):
        winner = play_game(p1, p2, env, manager, encoder)
        if winner == 1: s1+=1; w1+=1; l2+=1
        elif winner == -1: s2+=1; w2+=1; l1+=1
        else: s1+=0.5; s2+=0.5; d1+=1; d2+=1

    # P2 is Red
    for i in range(GAMES_PER_MATCH // 2):
        winner = play_game(p2, p1, env, manager, encoder)
        # Note: play_game returns 1 if Red wins. Here Red is P2.
        if winner == 1: s2+=1; w2+=1; l1+=1
        elif winner == -1: s1+=1; w1+=1; l2+=1
        else: s1+=0.5; s2+=0.5; d1+=1; d2+=1
        
    print(f"\r{header:<30} | Final: {s1}-{s2} ({d1} Draws)")
    return (s1, w1, l1, d1), (s2, w2, l2, d2)

# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    manager = ActionManager(device=DEVICE)
    encoder = CheckersBoardEncoder()
    env = CheckersEnv()
    
    # Load Players
    players = []
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth"))
    
    for f in files:
        # Filter by iteration number
        try:
            iter_num = int(f.split("_")[-1].replace(".pth", ""))
            if TARGET_ITERS and iter_num not in TARGET_ITERS:
                continue
            
            p = load_player(f, manager)
            if p: players.append(p)
        except:
            continue
            
    players.sort(key=lambda x: int(x.name.split(" ")[1])) # Sort by iter
    
    print(f"\n🏆 ALPHAZERO TOURNAMENT ({len(players)} Players)")
    print(f"Sims: {MCTS_SIMS} | Games/Match: {GAMES_PER_MATCH}")
    
    matchups = list(itertools.combinations(players, 2))
    
    for p1, p2 in matchups:
        stats1, stats2 = run_match(p1, p2, env, manager, encoder)
        p1.update_stats(*stats1)
        p2.update_stats(*stats2)
        
    print("\n" + "="*60)
    print(f"{'RANK':<5} {'NAME':<15} {'PTS':<5} {'W-L-D':<10} {'WIN%':<5}")
    print("="*60)
    
    players.sort(key=lambda x: x.score, reverse=True)
    for i, p in enumerate(players):
        total = p.matches_played * GAMES_PER_MATCH
        wp = (p.score / total * 100) if total > 0 else 0
        print(f"{i+1:<5} {p.name:<15} {p.score:<5} {p.wins}-{p.losses}-{p.draws:<6} {wp:.1f}%")

if __name__ == "__main__":
    main()