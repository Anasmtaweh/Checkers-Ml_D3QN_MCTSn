#!/usr/bin/env python3
"""
round_robin_tournament.py - Full Round Robin Tournament (D3QN + AlphaZero MCTS)

What changed vs your original:
- Can load AlphaZero checkpoints (dict with 'model_state_dict') and play them using MCTS.
- Keeps D3QN + legacy DQN support.
- Uses cp == -1 canonicalization / action-id mapping for D3QN, same idea as web/app.py.

Format:
- Round Robin (All vs All).
- GAMES_PER_MATCH games per pairing (half as Red / half as Black).
- Mercy Rule: MAX_MOVES moves.

Author: ML Engineer
"""

import os
import sys
import glob
import itertools
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.game import CheckersEnv
from core.action_manager import ActionManager
from core.board_encoder import CheckersBoardEncoder
from core.move_parser import parse_legal_moves

from training.d3qn.model import D3QNModel
from training.alpha_zero.network import AlphaZeroModel
from training.alpha_zero.mcts import MCTS


# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

DIRECTORIES_TO_SCAN = [
    # AlphaZero checkpoints
    os.path.join(PROJECT_ROOT, "checkpoints", "alphazero"),
    # D3QN agents
    os.path.join(PROJECT_ROOT, "agents", "d3qn"),

    # Keep your original extra folder (harmless if empty / no .pth)
    os.path.join(PROJECT_ROOT, "data", "tournament_results"),
]

GAMES_PER_MATCH = 10      # Total games per match (5 as P1, 5 as P2)
MAX_MOVES = 550           # Professional checkers standard
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MCTS settings (used only for AlphaZeroModel agents)
USE_MCTS_FOR_ALPHazero = True
MCTS_SIMS = 400
MCTS_TEMP = 0.25
MCTS_C_PUCT = 1.5


# ════════════════════════════════════════════════════════════════════
# MODELS
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
        if x.dim() == 3:
            x = x.unsqueeze(0)
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
        return self.forward(state)


class Player:
    def __init__(self, name, path, model, kind):
        self.name = name
        self.path = path
        self.model = model
        self.kind = kind  # "mcts" | "dqn" | "legacy"
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
    """
    Differentiates MCTS (AlphaZero) vs DQN based on checkpoint signature.
    - AlphaZero: dict with 'model_state_dict' -> Player.kind = 'mcts'
    - D3QN: dict with 'model_online'/'online'/... or raw state dict -> Player.kind = 'dqn'
    - Legacy fallback -> Player.kind = 'legacy'
    """
    try:
        checkpoint = torch.load(path, map_location=DEVICE)

        # --- MCTS / AlphaZero ---
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model = AlphaZeroModel(action_dim=action_manager.action_dim, device=DEVICE)
            model.network.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            return Player(name, path, model, kind="mcts")  # <-- explicit

        # --- DQN / Legacy ---
        if isinstance(checkpoint, dict):
            if "model_online" in checkpoint:
                state_dict = checkpoint["model_online"]
            elif "online" in checkpoint:
                state_dict = checkpoint["online"]
            elif "online_model_state_dict" in checkpoint:
                state_dict = checkpoint["online_model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Try Gen12+ dual-head D3QNModel
        model = D3QNModel(action_manager.action_dim, DEVICE).to(DEVICE)
        try:
            model.online.load_state_dict(state_dict)
            model.eval()
            return Player(name, path, model, kind="dqn")
        except RuntimeError:
            legacy = LegacyDuelingDQN(action_manager.action_dim, DEVICE).to(DEVICE)
            legacy.load_state_dict(state_dict)
            legacy.eval()
            return Player(name, path, legacy, kind="legacy")

    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
        return None


# ════════════════════════════════════════════════════════════════════
# MOVE SELECTION
# ════════════════════════════════════════════════════════════════════

def pick_legal_by_pair(legal_moves, action_manager, move_pair):
    if move_pair is None:
        return None
    for m in legal_moves:
        if action_manager._extract_start_landing(m) == move_pair:
            return m
    return None
def get_greedy_dqn_move(model, env, action_manager, encoder):
    legal = env.get_legal_moves()
    if not legal:
        return None
    
    cp = env.current_player
    state = encoder.encode(env.board.get_state(), cp).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if hasattr(model, "get_q_values"):
            q = model.get_q_values(state, player_side=cp)
        elif hasattr(model, "online"):
            q = model.online(state)
        else:
            q = model(state)

    # Masking:
    # For cp == -1, canonicalize legal moves via flip to match action ids, then map back.
    if cp == -1:
        norm, _ = parse_legal_moves(legal, action_manager)
        can = [action_manager.flip_move(m) for m in norm]
        mask = action_manager.make_legal_action_mask(can).to(DEVICE)

        can_to_abs = {
            action_manager.get_action_id(c): action_manager.get_action_id(n)
            for c, n in zip(can, norm)
        }
    else:
        mask = action_manager.make_legal_action_mask(legal).to(DEVICE)
        can_to_abs = None

    q = q.clone()
    q[0, ~mask] = -float("inf")
    aid = int(torch.argmax(q, dim=1).item())

    if cp == -1 and can_to_abs is not None:
        aid = can_to_abs.get(aid, aid)

    move_pair = action_manager.get_move_from_id(aid)
    selected = pick_legal_by_pair(legal, action_manager, move_pair)

    # Safety fallback (if coords got flipped/canonicalized oddly)
    if selected is None and cp == -1:
        selected = pick_legal_by_pair(legal, action_manager, action_manager.flip_move(move_pair))

    if selected is None:
        selected = legal[0]

    return selected
def get_move_for_player(player, env, action_manager, encoder):
    legal = env.get_legal_moves()
    if not legal:
        return None

    cp = env.current_player

    # --- MCTS path ONLY for MCTS players ---
    if player.kind == "mcts":
        mcts = MCTS(
            player.model, action_manager, encoder,
            c_puct=MCTS_C_PUCT,
            num_simulations=MCTS_SIMS,
            device=DEVICE,
            dirichlet_alpha=0.0
        )
        sim_env = copy.deepcopy(env)
        probs, _root = mcts.get_action_prob(sim_env, temp=MCTS_TEMP, training=False)
        aid = int(np.argmax(probs))
        selected = mcts._get_move_from_action(aid, legal, player=cp)
        return selected if selected is not None else random.choice(legal)

    # --- Otherwise: DQN greedy path ---
    return get_greedy_dqn_move(player.model, env, action_manager, encoder)


# ════════════════════════════════════════════════════════════════════
# TOURNAMENT ENGINE
# ════════════════════════════════════════════════════════════════════

def play_game(env, p_red: Player, p_black: Player, manager, encoder):
    env.reset()
    done = False
    moves = 0

    while not done:
        moves += 1
        if moves > MAX_MOVES:
            # timeout draw
            return 0

        current = env.current_player
        player = p_red if current == 1 else p_black

        move = get_move_for_player(player, env, manager, encoder)
        if not move:
            # no move -> current player loses
            return -1 if current == 1 else 1

        _, _, done, info = env.step(move)
        if done:
            return info.get("winner", 0)

    return 0


def run_match(p1: Player, p2: Player, env, manager, encoder):
    header = f"   ⚔️  {p1.name:<20} vs {p2.name:<20}"
    print(f"{header} | Starting...", end="", flush=True)

    s1, s2 = 0.0, 0.0
    w1, l1, d1 = 0, 0, 0
    w2, l2, d2 = 0, 0, 0

    games_p1_red = GAMES_PER_MATCH // 2
    games_p2_red = GAMES_PER_MATCH - games_p1_red

    # p1 is Red
    for i in range(games_p1_red):
        r = play_game(env, p1, p2, manager, encoder)
        if r == 1:
            s1 += 1.0; w1 += 1; l2 += 1
        elif r == -1:
            s2 += 1.0; l1 += 1; w2 += 1
        else:
            s1 += 0.5; s2 += 0.5; d1 += 1; d2 += 1

        print(f"\r{header} | Game {i+1}/{GAMES_PER_MATCH} | Score: {s1:.1f}-{s2:.1f}", end="", flush=True)

    # p2 is Red
    for i in range(games_p2_red):
        r = play_game(env, p2, p1, manager, encoder)
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
    print("=" * 80)
    print("🏆 ROUND ROBIN TOURNAMENT 🏆")
    print("=" * 80)

    manager = ActionManager(device=DEVICE)
    encoder = CheckersBoardEncoder()
    env = CheckersEnv()

    players = []
    seen_names = set()

    print("Scanning directories...")
    for directory in DIRECTORIES_TO_SCAN:
        if not os.path.exists(directory):
            print(f"⚠️ Directory not found: {directory}")
            continue

        files = glob.glob(os.path.join(directory, "*.pth"))
        print(f"  Found {len(files)} agents in '{directory}'")

        for f in files:
            name = os.path.basename(f).replace(".pth", "")
            if name in seen_names:
                continue

            agent = load_agent(name, f, manager)
            if agent is not None:
                players.append(agent)
                seen_names.add(name)

    if len(players) < 2:
        print("Not enough players found!")
        return

    print(f"\n✅ TOURNAMENT READY: {len(players)} Competitors")
    print(f"Format: Round Robin x {GAMES_PER_MATCH} Games per Match")
    print(f"Device: {DEVICE}")
    print(f"MCTS: {'ON' if USE_MCTS_FOR_ALPHazero else 'OFF'} | sims={MCTS_SIMS} temp={MCTS_TEMP}\n")

    matchups = list(itertools.combinations(players, 2))
    total_matches = len(matchups)

    print(f"Total Matches Scheduled: {total_matches}")
    print("-" * 80)

    for idx, (p1, p2) in enumerate(matchups):
        print(f"\nMatch {idx+1}/{total_matches}")
        stats1, stats2 = run_match(p1, p2, env, manager, encoder)
        p1.update_stats(*stats1)
        p2.update_stats(*stats2)

    print("\n" + "=" * 80)
    print("🏅 FINAL LEADERBOARD")
    print("=" * 80)
    print(f"{'RK':<3} | {'NAME':<30} | {'KIND':<9} | {'PTS':<6} | {'W':<4} | {'L':<4} | {'D':<4} | {'WIN RATE'}")
    print("-" * 80)

    players.sort(key=lambda x: x.score, reverse=True)

    for i, p in enumerate(players):
        max_pts = p.matches_played * GAMES_PER_MATCH
        pct = (p.score / max_pts * 100) if max_pts > 0 else 0.0
        print(f"{i+1:<3} | {p.name:<30} | {p.kind:<9} | {p.score:<6.1f} | {p.wins:<4} | {p.losses:<4} | {p.draws:<4} | {pct:.1f}%")


if __name__ == "__main__":
    main()