#!/usr/bin/env python3
"""
iron_tournament_ultimate.py - THE ROYAL RUMBLE

Purpose:
  Finds EVERY .pth file in 'checkpoints_iron_league_v3' AND 'opponent_pool'.
  Runs a massive Swiss Tournament to rank them all from best to worst.

Format:
  - Swiss System (Winners play Winners).
  - 10 Games per Match (5 Red / 5 Black).
  - Mercy Rule: 200 Moves.

Author: ML Engineer
Date: December 20, 2025
"""

import torch
import numpy as np
import os
import glob
from checkers_env.env import CheckersEnv
from training.common.action_manager import ActionManager
from training.common.board_encoder import CheckersBoardEncoder
from training.d3qn.model import D3QNModel

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

DIRECTORIES_TO_SCAN = [
    "checkpoints_iron_league_v3",  # Your new agents (500, 1000, 1500...)
    "opponent_pool"                # Your old agents (Gen 7, Glass Cannon...)
]

ROUNDS = 6               # Sufficient to rank 10-20 agents
PAIRS_PER_MATCH = 5      # 5 Pairs = 10 Games total per match
MAX_MOVES = 200          # Mercy rule
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ════════════════════════════════════════════════════════════════════
# TOURNAMENT ENGINE
# ════════════════════════════════════════════════════════════════════

class Player:
    def __init__(self, name, path, model):
        self.name = name
        self.path = path
        self.model = model
        self.score = 0.0
        self.matches_played = 0
        self.opponents = []
        self.buchholz = 0.0

    def update_score(self, points):
        self.score += points
        self.matches_played += 1

def load_agent(name, path, action_manager):
    """Universal Loader: Handles Checkpoints AND Raw State Dicts."""
    try:
        model = D3QNModel(action_manager.action_dim, DEVICE).to(DEVICE)
        checkpoint = torch.load(path, map_location=DEVICE)
        
        # Case 1: Full Checkpoint (New Iron Agent)
        if isinstance(checkpoint, dict) and 'model_online' in checkpoint:
            model.online.load_state_dict(checkpoint['model_online'])
        # Case 2: Raw State Dict (Old Pool Agents)
        else:
            model.online.load_state_dict(checkpoint)
        
        model.eval()
        return Player(name, path, model)
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
        return None

def get_greedy_move(model, env, action_manager):
    legal_moves = env.get_legal_moves()
    if not legal_moves: return None
    
    state_tensor = CheckersBoardEncoder().encode(env.board.get_state(), env.current_player).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        q_values = model.get_q_values(state_tensor)[0]
    
    mask = action_manager.make_legal_action_mask(legal_moves).to(DEVICE)
    q_values[~mask] = -float('inf')
    
    best_action_id = int(q_values.argmax().item())
    move_struct = action_manager.get_move_from_id(best_action_id)
    
    for lm in legal_moves:
        if isinstance(lm, list):
             if (tuple(lm[0][0]), tuple(lm[-1][1])) == move_struct: return lm
        elif len(lm) == 2:
             if (tuple(lm[0]), tuple(lm[1])) == move_struct: return lm
    return legal_moves[0]

def play_game(env, p1, p2, manager):
    state = env.reset()
    done = False
    moves = 0
    while not done:
        moves += 1
        if moves > MAX_MOVES: return 0 # Draw
        current = env.current_player
        move = get_greedy_move(p1 if current == 1 else p2, env, manager)
        if not move: return -1 if current == 1 else 1
        _, _, done, info = env.step(move)
        if done: return info.get('winner', 0)

def run_match(p1, p2, env, manager):
    print(f"   ⚔️  {p1.name:<25} vs {p2.name:<25}", end="", flush=True)
    s1, s2 = 0.0, 0.0
    for _ in range(PAIRS_PER_MATCH):
        # Game 1: P1 Red
        r = play_game(env, p1.model, p2.model, manager)
        s1 += (1.0 if r==1 else 0.5 if r==0 else 0)
        s2 += (1.0 if r==-1 else 0.5 if r==0 else 0)
        # Game 2: P2 Red
        r = play_game(env, p2.model, p1.model, manager)
        s2 += (1.0 if r==1 else 0.5 if r==0 else 0)
        s1 += (1.0 if r==-1 else 0.5 if r==0 else 0)
    print(f" | Score: {s1}-{s2}")
    return s1, s2

def main():
    print("="*80)
    print("🏆 ULTIMATE IRON TOURNAMENT (ALL AGENTS) 🏆")
    print("="*80)
    
    manager = ActionManager(device=DEVICE)
    env = CheckersEnv()
    players = []
    seen_names = set()

    # 1. SCAN EVERY FOLDER
    print("Scanning directories...")
    for directory in DIRECTORIES_TO_SCAN:
        search_path = os.path.join(directory, "*.pth")
        files = glob.glob(search_path)
        print(f"  Found {len(files)} agents in '{directory}'")
        
        for f in files:
            name = os.path.basename(f).replace(".pth", "")
            if name in seen_names: continue # Avoid duplicates
            
            agent = load_agent(name, f, manager)
            if agent:
                players.append(agent)
                seen_names.add(name)

    if len(players) < 2:
        print("Not enough players found!")
        return

    print(f"\n✅ TOURNAMENT READY: {len(players)} Competitors")
    print(f"Format: {ROUNDS} Rounds x {PAIRS_PER_MATCH*2} Games\n")

    # 2. RUN ROUNDS
    for r in range(1, ROUNDS + 1):
        print(f"\n🔔 ROUND {r}")
        print("-" * 80)
        players.sort(key=lambda x: x.score, reverse=True)
        
        paired = []
        skip = set()
        
        for i in range(len(players)):
            if i in skip: continue
            
            # Find best available opponent
            opponent_idx = -1
            for j in range(i+1, len(players)):
                if j not in skip:
                    opponent_idx = j
                    break
            
            if opponent_idx != -1:
                p1, p2 = players[i], players[opponent_idx]
                paired.append((p1, p2))
                skip.add(i)
                skip.add(opponent_idx)
            else:
                print(f"   ⚠️  {players[i].name} gets a BYE")
                players[i].update_score(float(PAIRS_PER_MATCH)) # Bye points (half max)

        for p1, p2 in paired:
            s1, s2 = run_match(p1, p2, env, manager)
            p1.update_score(s1)
            p2.update_score(s2)
            p1.opponents.append(p2)
            p2.opponents.append(p1)

    # 3. FINAL RESULTS
    print("\n" + "="*80)
    print("🏅 FINAL LEADERBOARD")
    print("="*80)
    print(f"{'RK':<3} | {'NAME':<30} | {'PTS':<6} | {'BHZ':<6} | {'WIN RATE'}")
    print("-" * 80)
    
    # Calc Buchholz & Sort
    for p in players: p.buchholz = sum(op.score for op in p.opponents)
    players.sort(key=lambda x: (x.score, x.buchholz), reverse=True)
    
    for i, p in enumerate(players):
        max_pts = p.matches_played * (PAIRS_PER_MATCH * 2)
        pct = (p.score / max_pts * 100) if max_pts > 0 else 0.0
        print(f"{i+1:<3} | {p.name:<30} | {p.score:<6.1f} | {p.buchholz:<6.1f} | {pct:.1f}%")

if __name__ == "__main__":
    main()