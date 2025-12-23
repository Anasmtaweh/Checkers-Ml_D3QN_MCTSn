#!/usr/bin/env python3
"""
MCTS Gauntlet: MCTS Agent vs All Saved Models
"""
import sys
import os
import copy
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.mcts.mcts_agent import MCTSAgent
from core.game import CheckersEnv
from core.action_manager import ActionManager
from core.board_encoder import CheckersBoardEncoder
from training.d3qn.model import D3QNModel

class CheckersGame:
    """
    Wrapper for CheckersEnv to be compatible with MCTS.
    """
    def __init__(self, env=None):
        if env:
            self.env = env
        else:
            self.env = CheckersEnv()
            self.env.reset()

    @property
    def current_player(self):
        return self.env.current_player

    @property
    def board(self):
        return self.env.board.board

    def get_valid_moves(self) -> List[Tuple]:
        """Return list of valid moves: [(from_pos, to_pos), ...]"""
        return self.env.get_legal_moves()
    
    def make_move(self, action: Tuple):
        """Execute a move"""
        self.env.step(action)
    
    def is_game_over(self) -> bool:
        """Check if game is finished"""
        return len(self.env.get_legal_moves()) == 0
    
    def get_winner(self) -> int:
        """Return: 1 (red wins), -1 (black wins), 0 (draw)"""
        if not self.is_game_over():
            return 0
        # If current player cannot move, they lose.
        return -1 if self.env.current_player == 1 else 1
    
    def clone(self):
        """Return a deep copy of the game state"""
        return CheckersGame(copy.deepcopy(self.env))

class LegacyDuelingDQN(nn.Module):
    """Fallback for older single-head models."""
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
        return self.forward(state)

def load_model(path, action_manager, device):
    """Load model from path, handling legacy architectures."""
    try:
        # Try loading as Gen 12 (Dual Head)
        model = D3QNModel(action_manager.action_dim, device).to(device)
        checkpoint = torch.load(path, map_location=device)
        
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
            # Fallback to Legacy
            model = LegacyDuelingDQN(action_manager.action_dim, device).to(device)
            model.load_state_dict(state_dict)
        
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Failed to load {os.path.basename(path)}: {e}")
        return None

def get_model_move(model, game, action_manager, device):
    """Get greedy move from D3QN model."""
    env = game.env
    legal_moves = env.get_legal_moves()
    if not legal_moves: return None
    
    state_tensor = CheckersBoardEncoder().encode(env.board.get_state(), env.current_player).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Handle both legacy (no player_side) and new models
        if isinstance(model, D3QNModel):
            q_values = model.get_q_values(state_tensor, player_side=env.current_player)[0]
        else:
            q_values = model.get_q_values(state_tensor)[0]
    
    mask = action_manager.make_legal_action_mask(legal_moves).to(device)
    q_values[~mask] = -float('inf')
    
    best_action_id = int(q_values.argmax().item())
    move_struct = action_manager.get_move_from_id(best_action_id)
    
    for lm in legal_moves:
        if isinstance(lm, list):
             if (tuple(lm[0][0]), tuple(lm[-1][1])) == move_struct: return lm
        elif len(lm) == 2:
             if (tuple(lm[0]), tuple(lm[1])) == move_struct: return lm
    return legal_moves[0]

def normalize_action(action):
    """Convert lists in action to tuples recursively to satisfy env.step typing."""
    if isinstance(action, tuple):
        return tuple(normalize_action(a) for a in action)
    if isinstance(action, list):
        return tuple(normalize_action(a) for a in action)
    return action

def play_match(model_name, model, mcts_agent, action_manager, device):
    """Play one game: Model (Red) vs MCTS (Black)."""
    game = CheckersGame()
    print(f"\n⚔️  Match: {model_name} (Red) vs MCTS (Black)")
    
    while not game.is_game_over():
        if game.current_player == 1:
            # Model Turn (Red)
            move = get_model_move(model, game, action_manager, device)
            if move:
                game.make_move(normalize_action(move))
            else:
                break
        else:
            # MCTS Turn (Black)
            move = mcts_agent.get_action(game)
            if move:
                game.make_move(normalize_action(move))
            else:
                break
    
    winner = game.get_winner()
    if winner == 1:
        print(f"   🏆 Winner: {model_name} (Red)")
        return 1
    elif winner == -1:
        print(f"   🤖 Winner: MCTS (Black)")
        return -1
    else:
        print(f"   🤝 Draw")
        return 0

def run_mcts_gauntlet():
    print("=" * 60)
    print("CHECKERS - MCTS GAUNTLET")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    action_manager = ActionManager(device=device)
    
    # Load a strong evaluator model (use the champion)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    champion_path = os.path.join(project_root, "agents/d3qn", "DQN_CHAMPION_ep500_62pct_tournament.pth")
    eval_model = None
    if os.path.exists(champion_path):
        eval_model = load_model(champion_path, action_manager, device)
        print(f"✓ Loaded champion model as neural evaluator")
    
    # 7 seconds per move - balance between strength and reasonable play time
    mcts = MCTSAgent(time_limit=7.0, exploration_weight=2.0, eval_model=eval_model, action_manager=action_manager, device=device)
    
    # Directories to scan
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    dirs = [
        os.path.join(project_root, "agents/d3qn"),
        os.path.join(project_root, "data/archives"),
    ]
    
    models_found = []
    for d in dirs:
        if os.path.exists(d):
            files = glob.glob(os.path.join(d, "*.pth"))
            models_found.extend(files)
    
    # Sort to play in order
    models_found = sorted(list(set(models_found)))
    print(f"Found {len(models_found)} models to challenge.")
    
    mcts_wins = 0
    model_wins = 0
    draws = 0
    
    for path in models_found:
        name = os.path.basename(path).replace(".pth", "")
        model = load_model(path, action_manager, device)
        if not model: continue
        
        result = play_match(name, model, mcts, action_manager, device)
        if result == 1: model_wins += 1
        elif result == -1: mcts_wins += 1
        else: draws += 1
        
    print("\n" + "=" * 60)
    print("GAUNTLET RESULTS")
    print(f"Models Defeated: {model_wins}")
    print(f"MCTS Wins:       {mcts_wins}")
    print(f"Draws:           {draws}")
    print("=" * 60)

if __name__ == "__main__":
    run_mcts_gauntlet()
