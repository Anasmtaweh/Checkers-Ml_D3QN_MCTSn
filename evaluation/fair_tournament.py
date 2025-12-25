import os
import sys
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.game import CheckersEnv
from core.action_manager import ActionManager
from core.board_encoder import CheckersBoardEncoder
from core.move_parser import parse_legal_moves
from training.d3qn.model import DuelingDQN


class OldDuelingDQN(torch.nn.Module):
    """Old single-head Dueling DQN architecture (pre-Gen12)"""
    def __init__(self, action_dim: int, device="cpu"):
        super(OldDuelingDQN, self).__init__()
        self.action_dim = action_dim
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # CNN Backbone
        self.conv1 = torch.nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.flatten_size = 64 * 8 * 8
        
        # Layer norm (present in old models)
        self.fc_norm = torch.nn.LayerNorm(self.flatten_size)
        
        # FC layers (single head, 512 hidden units)
        self.value_fc1 = torch.nn.Linear(self.flatten_size, 512)
        self.value_fc2 = torch.nn.Linear(512, 1)
        self.advantage_fc1 = torch.nn.Linear(self.flatten_size, 512)
        self.advantage_fc2 = torch.nn.Linear(512, action_dim)
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor, player_side: int = 1) -> torch.Tensor:
        """Forward pass - ignores player_side for compatibility"""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc_norm(x)
        
        # Value stream
        value = torch.nn.functional.relu(self.value_fc1(x))
        value = self.value_fc2(value)
        
        # Advantage stream
        advantage = torch.nn.functional.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)
        
        # Dueling aggregation: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class FairTournament:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.env = CheckersEnv()
        self.action_manager = ActionManager(device=device)
        self.encoder = CheckersBoardEncoder()
        self.models = {}
        self.results = defaultdict(lambda: {
            "p1_wins": 0, "p1_losses": 0, "p1_draws": 0,
            "p2_wins": 0, "p2_losses": 0, "p2_draws": 0,
            "games_played": 0
        })
        self.matchups = []
        
    def load_all_agents(self, agent_dirs):
        """Load all .pth models from specified directories, auto-detecting architecture"""
        for agent_dir in agent_dirs:
            if not os.path.exists(agent_dir):
                continue
            
            for model_file in Path(agent_dir).glob("*.pth"):
                agent_name = model_file.stem
                try:
                    checkpoint = torch.load(str(model_file), map_location=self.device)
                    
                    # Extract actual state dict from checkpoint
                    if isinstance(checkpoint, dict):
                        if "model_online" in checkpoint:
                            state_dict = checkpoint["model_online"]
                        elif "model_state_dict" in checkpoint:
                            state_dict = checkpoint["model_state_dict"]
                        elif "state_dict" in checkpoint:
                            state_dict = checkpoint["state_dict"]
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint
                    
                    # Detect architecture from state_dict keys
                    if "p1_value_fc1.weight" in state_dict:
                        # New architecture: P1/P2 dual-head
                        # Detect action_dim from advantage layer
                        action_dim = state_dict["p1_advantage_fc2.weight"].shape[0]
                        model = DuelingDQN(action_dim=action_dim, device=self.device).to(self.device)
                        model.load_state_dict(state_dict)
                        model.eval()
                        self.models[agent_name] = model
                        print(f"✓ Loaded (new, {action_dim} actions): {agent_name}")
                    
                    elif "value_fc1.weight" in state_dict:
                        # Old architecture: single-head
                        action_dim = state_dict["advantage_fc2.weight"].shape[0]
                        model = OldDuelingDQN(action_dim=action_dim, device=self.device)
                        model.load_state_dict(state_dict)
                        model.eval()
                        self.models[agent_name] = model
                        print(f"✓ Loaded (old, {action_dim} actions): {agent_name}")
                    else:
                        print(f"✗ Unknown architecture: {agent_name}")
                
                except Exception as e:
                    print(f"✗ Failed to load {agent_name}: {e}")
    
    def get_best_action(self, agent_model, legal_moves, current_player):
        """Get deterministic best action (no exploration)"""
        state_tensor = self.encoder.encode(
            self.env.board.get_state(), 
            current_player
        ).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Handle both old and new model architectures
            q_values = agent_model(state_tensor, player_side=1 if current_player == 1 else -1)
        
        # --- FIX FOR P2 PERSPECTIVE ---
        if current_player == -1:
            # 1. Flip legal moves to Canonical (P1) perspective
            #    We need to parse them first because legal_moves contains raw env moves (lists, etc)
            normalized_moves, mapping = parse_legal_moves(legal_moves, self.action_manager)
            
            canonical_moves = [self.action_manager.flip_move(m) for m in normalized_moves]
            mask = self.action_manager.make_legal_action_mask(canonical_moves).to(self.device)
            
            # Map Canonical ID -> Absolute ID
            canonical_to_absolute = {}
            for i, cm in enumerate(canonical_moves):
                cid = self.action_manager.get_action_id(cm)
                if cid >= 0:
                    orig_move = normalized_moves[i]
                    aid = self.action_manager.get_action_id(orig_move)
                    canonical_to_absolute[cid] = aid
        else:
            # P1: Canonical = Absolute
            mask = self.action_manager.make_legal_action_mask(legal_moves).to(self.device)
            canonical_to_absolute = None
        
        # Mask illegal moves
        q_values[0, ~mask] = -float('inf')
        
        best_action_id = int(q_values.argmax().item())
        
        # If P2, best_action_id is Canonical. We need to map it back to Absolute.
        if current_player == -1 and canonical_to_absolute is not None:
             best_action_id = canonical_to_absolute.get(best_action_id, -1)
        
        # Convert back to move using action_manager
        best_move_struct = self.action_manager.get_move_from_id(best_action_id)
        
        # Find the corresponding move in legal_moves
        for move in legal_moves:
            # Normalize move to structure ((r1,c1), (r2,c2)) for comparison
            if isinstance(move, list):
                # Capture chain: check start and end
                if (tuple(move[0][0]), tuple(move[-1][1])) == best_move_struct:
                    return move
            elif isinstance(move, tuple) and len(move) == 2:
                # Simple move
                if (tuple(move[0]), tuple(move[1])) == best_move_struct:
                    return move
        
        # Fallback: return first legal move
        return legal_moves[0] if legal_moves else None
    
    def play_game(self, agent_p1, agent_p2, max_steps=200):
        """Play one complete game. Returns winner (1, 2, or 0 for draw)"""
        self.env.reset()
        
        for step in range(max_steps):
            current_player = self.env.current_player
            legal_moves = self.env.get_legal_moves()
            
            if not legal_moves:
                # Current player has no moves = loses
                # If P1 (1) loses, winner is P2 (2)
                # If P2 (-1) loses, winner is P1 (1)
                return 2 if current_player == 1 else 1
            
            # Get move from appropriate agent
            if current_player == 1:
                move = self.get_best_action(self.models[agent_p1], legal_moves, current_player)
            else:
                move = self.get_best_action(self.models[agent_p2], legal_moves, current_player)
            
            if move is None:
                return 2 if current_player == 1 else 1
            
            state, reward, done, info = self.env.step(move)
            
            if done:
                winner = info.get("winner", 0)
                if winner == 1: return 1
                elif winner == -1: return 2
                else: return 0
        
        # Max steps reached = draw
        return 0
    
    def run_tournament(self, games_per_matchup=2):
        """
        Run full round-robin tournament.
        games_per_matchup=2: each pair plays twice (once as P1, once as P2)
        """
        agent_names = list(self.models.keys())
        total_games = len(agent_names) * (len(agent_names) - 1) * games_per_matchup
        game_count = 0
        
        print(f"\n{'='*80}")
        print(f"FAIR TOURNAMENT: {len(agent_names)} agents")
        print(f"Total games: {total_games}")
        print(f"{'='*80}\n")
        
        # Each pair plays twice (A vs B as P1/P2, then B vs A as P1/P2)
        for i, agent_a in enumerate(agent_names):
            for j, agent_b in enumerate(agent_names):
                if i == j:  # Skip self-play
                    continue
                
                for game_num in range(games_per_matchup):
                    game_count += 1
                    winner = self.play_game(agent_a, agent_b)
                    
                    # Update results
                    if winner == 1:  # P1 wins
                        self.results[agent_a]["p1_wins"] += 1
                        self.results[agent_b]["p2_losses"] += 1
                    elif winner == 2:  # P2 wins
                        self.results[agent_a]["p1_losses"] += 1
                        self.results[agent_b]["p2_wins"] += 1
                    else:  # Draw
                        self.results[agent_a]["p1_draws"] += 1
                        self.results[agent_b]["p2_draws"] += 1
                    
                    self.results[agent_a]["games_played"] += 1
                    self.results[agent_b]["games_played"] += 1
                    
                    # Progress
                    if game_count % 10 == 0:
                        print(f"Progress: {game_count}/{total_games} games completed")
    
    def calculate_stats(self):
        """Calculate win rates and stats for each agent"""
        stats = {}
        for agent_name, results in self.results.items():
            p1_total = results["p1_wins"] + results["p1_losses"] + results["p1_draws"]
            p2_total = results["p2_wins"] + results["p2_losses"] + results["p2_draws"]
            total = results["games_played"]
            
            p1_wr = (results["p1_wins"] / p1_total * 100) if p1_total > 0 else 0
            p2_wr = (results["p2_wins"] / p2_total * 100) if p2_total > 0 else 0
            overall_wr = ((results["p1_wins"] + results["p2_wins"]) / total * 100) if total > 0 else 0
            
            stats[agent_name] = {
                "p1_wins": results["p1_wins"],
                "p1_losses": results["p1_losses"],
                "p1_draws": results["p1_draws"],
                "p1_wr": p1_wr,
                "p2_wins": results["p2_wins"],
                "p2_losses": results["p2_losses"],
                "p2_draws": results["p2_draws"],
                "p2_wr": p2_wr,
                "total_games": total,
                "overall_wr": overall_wr,
                "p1_p2_balance": abs(p1_wr - p2_wr)
            }
        
        return stats
    
    def print_results(self):
        """Print tournament results in readable format"""
        stats = self.calculate_stats()
        
        # Sort by overall win rate
        sorted_stats = sorted(
            stats.items(), 
            key=lambda x: x[1]["overall_wr"], 
            reverse=True
        )
        
        print(f"\n{'='*100}")
        print("TOURNAMENT RESULTS")
        print(f"{'='*100}\n")
        
        print(f"{'Rank':<6}{'Agent':<30}{'P1 WR':<12}{'P2 WR':<12}{'Overall':<12}{'Balance':<12}{'Games':<8}")
        print("-" * 100)
        
        for rank, (agent_name, stat) in enumerate(sorted_stats, 1):
            print(
                f"{rank:<6}"
                f"{agent_name:<30}"
                f"{stat['p1_wr']:.1f}%{'':<6}"
                f"{stat['p2_wr']:.1f}%{'':<6}"
                f"{stat['overall_wr']:.1f}%{'':<6}"
                f"{stat['p1_p2_balance']:.1f}%{'':<6}"
                f"{stat['total_games']:<8}"
            )
        
        print("\n" + "="*100)
        print("DETAILED BREAKDOWN")
        print("="*100 + "\n")
        
        for rank, (agent_name, stat) in enumerate(sorted_stats, 1):
            print(f"\n{rank}. {agent_name}")
            print(f"   P1: {stat['p1_wins']}W - {stat['p1_losses']}L - {stat['p1_draws']}D (WR: {stat['p1_wr']:.1f}%)")
            print(f"   P2: {stat['p2_wins']}W - {stat['p2_losses']}L - {stat['p2_draws']}D (WR: {stat['p2_wr']:.1f}%)")
            print(f"   Overall: {stat['overall_wr']:.1f}% | P1/P2 Balance Gap: {stat['p1_p2_balance']:.1f}%")
    
    def save_results(self, output_file="tournament_results.json"):
        """Save results to JSON"""
        stats = self.calculate_stats()
        
        output = {
            "timestamp": datetime.now().isoformat(),
            "total_agents": len(self.models),
            "results": stats
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a fair Round-Robin Checkers Tournament.")
    parser.add_argument("--games", type=int, default=10, help="Games per specific matchup (A vs B). Total games per pair will be 2x this (A as P1 + A as P2).")
    parser.add_argument("--output", type=str, default="fair_tournament_results.json", help="Output JSON file for results.")
    args = parser.parse_args()

    # Scan these directories for all .pth models
    agent_dirs = [
        "/home/anas/ML_Gen2/agents/d3qn",
        "/home/anas/ML_Gen2/checkpoints_gen11_decisive",
        "/home/anas/ML_Gen2/checkpoints_gen12_elite",
        "/home/anas/ML_Gen2/checkpoints_iron_league_v3",
        "/home/anas/ML_Gen2/gen12_elite_3500",
    ]
    
    tournament = FairTournament()
    tournament.load_all_agents(agent_dirs)
    
    num_agents = len(tournament.models)
    if num_agents > 1:
        # Calculate total games per agent
        # (N-1) opponents * 2 roles (P1/P2) * games_per_matchup
        total_per_agent = (num_agents - 1) * 2 * args.games
        print(f"\nConfiguration:")
        print(f"  • Agents: {num_agents}")
        print(f"  • Games per matchup: {args.games}")
        print(f"  • Est. Total Games per Agent: {total_per_agent}")
        print(f"  • Margin of Error (95% CI): ±{100 / (total_per_agent**0.5):.1f}% (approx)\n")

        # Run tournament
        tournament.run_tournament(games_per_matchup=args.games)
        tournament.print_results()
        tournament.save_results(args.output)
    else:
        print("❌ Need at least 2 agents to run tournament")