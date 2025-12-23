#!/usr/bin/env python3
"""
round_robin_tournament.py - Full Round Robin Tournament

Purpose:
  Runs a full round-robin tournament between all agents found in specified directories.
  Every agent plays every other agent.

Directories:
  - opponent_pool
  - final_models

Format:
  - Round Robin (All vs All).
  - 10 Games per Match (5 Red / 5 Black).
  - Mercy Rule: 200 Moves.

Author: ML Engineer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import sys
import itertools

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.game import CheckersEnv
from core.action_manager import ActionManager
from core.board_encoder import CheckersBoardEncoder
from training.d3qn.model import D3QNModel

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

DIRECTORIES_TO_SCAN = [
    os.path.join(PROJECT_ROOT, "agents/d3qn"),
    os.path.join(PROJECT_ROOT, "data/tournament_results"),
]

GAMES_PER_MATCH = 10     # Total games per match (5 as P1, 5 as P2)
MAX_MOVES = 550         # Professional checkers standard
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ════════════════════════════════════════════════════════════════════
# TOURNAMENT ENGINE
# ════════════════════════════════════════════════════════════════════

class LegacyDuelingDQN(nn.Module):
    """
    Fallback architecture for Gen 11 and older agents (Single Head).
    Used when loading fails for the new Dual-Head D3QNModel.
    """
    def __init__(self, action_dim, device):
        super(LegacyDuelingDQN, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(5, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.flatten_size = 64 * 8 * 8
        self.fc_norm = nn.LayerNorm(self.flatten_size)
        self.value_fc1 = nn.Linear(self.flatten_size, 512)
        self.value_fc2 = nn.Linear(512, 1)
        self.advantage_fc1 = nn.Linear(self.flatten_size, 512)
        self.advantage_fc2 = nn.Linear(512, action_dim)
        self.to(device)

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(0)
        x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc_norm(x)
        val = self.value_fc2(F.relu(self.value_fc1(x)))
        adv = self.advantage_fc2(F.relu(self.advantage_fc1(x)))
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def get_q_values(self, state, player_side=1):
        # Legacy models ignore player_side (single head)
        return self.forward(state)

class Player:
    def __init__(self, name, path, model):
        self.name = name
        self.path = path
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

def load_agent(name, path, action_manager):
    """Universal Loader: Handles Checkpoints AND Raw State Dicts."""
    try:
        # 1. Try loading as Gen 12 (Dual Head)
        model = D3QNModel(action_manager.action_dim, DEVICE).to(DEVICE)
        checkpoint = torch.load(path, map_location=DEVICE)
        
        state_dict = None
        if isinstance(checkpoint, dict):
            if 'model_online' in checkpoint: state_dict = checkpoint['model_online']
            elif 'online' in checkpoint: state_dict = checkpoint['online']
            elif 'online_model_state_dict' in checkpoint: state_dict = checkpoint['online_model_state_dict']
            else: state_dict = checkpoint
        else:
            state_dict = checkpoint

        try:
            model.online.load_state_dict(state_dict)
        except RuntimeError:
            # 2. Fallback to Legacy Gen 11 (Single Head)
            # print(f"⚠️  Legacy model detected: {name}")
            model = LegacyDuelingDQN(action_manager.action_dim, DEVICE).to(DEVICE)
            model.load_state_dict(state_dict)
        
        if hasattr(model, 'eval'): model.eval()
        return Player(name, path, model)
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
        return None

def get_greedy_move(model, env, action_manager):
    legal_moves = env.get_legal_moves()
    if not legal_moves: return None
    
    state_tensor = CheckersBoardEncoder().encode(env.board.get_state(), env.current_player).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        # Pass player_side (1 or -1) for Gen 12 models to use correct head
        q_values = model.get_q_values(state_tensor, player_side=env.current_player)[0]
    
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
        if moves > MAX_MOVES:
            print(f"    ⏱️  TIMEOUT DRAW ({MAX_MOVES} moves)")
            return 0
        current = env.current_player
        move = get_greedy_move(p1 if current == 1 else p2, env, manager)
        if not move:
            print(f"  [Game finished in {moves} moves]")
            return -1 if current == 1 else 1
        _, _, done, info = env.step(move)
        if done:
            print(f"  [Game finished in {moves} moves]")
            return info.get('winner', 0)

def run_match(p1, p2, env, manager):
    header = f"   ⚔️  {p1.name:<20} vs {p2.name:<20}"
    print(f"{header} | Starting...", end="", flush=True)
    
    s1, s2 = 0.0, 0.0
    w1, l1, d1 = 0, 0, 0
    w2, l2, d2 = 0, 0, 0
    
    # Half games p1 is Red, Half p2 is Red
    games_p1_red = GAMES_PER_MATCH // 2
    games_p2_red = GAMES_PER_MATCH - games_p1_red
    
    for i in range(games_p1_red):
        r = play_game(env, p1.model, p2.model, manager)
        if r == 1:
            s1 += 1.0; w1 += 1; l2 += 1
        elif r == -1:
            s2 += 1.0; l1 += 1; w2 += 1
        else:
            s1 += 0.5; s2 += 0.5; d1 += 1; d2 += 1
        print(f"\r{header} | Game {i+1}/{GAMES_PER_MATCH} | Score: {s1:.1f}-{s2:.1f}", end="", flush=True)

    for i in range(games_p2_red):
        r = play_game(env, p2.model, p1.model, manager)
        if r == 1:
            s2 += 1.0; w2 += 1; l1 += 1
        elif r == -1:
            s1 += 1.0; w1 += 1; l2 += 1
        else:
            s1 += 0.5; s2 += 0.5; d1 += 1; d2 += 1
        print(f"\r{header} | Game {games_p1_red + i+1}/{GAMES_PER_MATCH} | Score: {s1:.1f}-{s2:.1f}", end="", flush=True)
        
    print(f"\r{header} | Final: {s1:.1f}-{s2:.1f} ({GAMES_PER_MATCH} games)        ")
    return (s1, w1, l1, d1), (s2, w2, l2, d2)

def main():
    print("="*80)
    print("🏆 ROUND ROBIN TOURNAMENT 🏆")
    print("="*80)
    
    manager = ActionManager(device=DEVICE)
    env = CheckersEnv()
    players = []
    seen_names = set()

    # 1. SCAN DIRECTORIES
    print("Scanning directories...")
    for directory in DIRECTORIES_TO_SCAN:
        if not os.path.exists(directory):
            print(f"⚠️ Directory not found: {directory}")
            continue
            
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
    print(f"Format: Round Robin x {GAMES_PER_MATCH} Games per Match\n")

    # 2. RUN MATCHES
    matchups = list(itertools.combinations(players, 2))
    total_matches = len(matchups)
    
    print(f"Total Matches Scheduled: {total_matches}")
    print("-" * 80)

    for idx, (p1, p2) in enumerate(matchups):
        print(f"\nMatch {idx+1}/{total_matches}")
        stats1, stats2 = run_match(p1, p2, env, manager)
        
        p1.update_stats(*stats1)
        p2.update_stats(*stats2)

    # 3. FINAL RESULTS
    print("\n" + "="*80)
    print("🏅 FINAL LEADERBOARD")
    print("="*80)
    print(f"{'RK':<3} | {'NAME':<30} | {'PTS':<6} | {'W':<4} | {'L':<4} | {'D':<4} | {'WIN RATE'}")
    print("-" * 80)
    
    players.sort(key=lambda x: x.score, reverse=True)
    
    for i, p in enumerate(players):
        max_pts = p.matches_played * GAMES_PER_MATCH
        pct = (p.score / max_pts * 100) if max_pts > 0 else 0.0
        print(f"{i+1:<3} | {p.name:<30} | {p.score:<6.1f} | {p.wins:<4} | {p.losses:<4} | {p.draws:<4} | {pct:.1f}%")

if __name__ == "__main__":
    main()