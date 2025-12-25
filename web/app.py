import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flask import Flask, render_template, jsonify, request

# Add parent directory to path so we can import your existing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.game import CheckersEnv
from core.action_manager import ActionManager
from core.board_encoder import CheckersBoardEncoder
from core.move_parser import parse_legal_moves
from training.d3qn.model import D3QNModel

# Fix for TemplateNotFound: Explicitly define paths relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = base_dir
static_dir = base_dir

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# --- GLOBAL GAME STATE ---
device = "cpu"  # CPU is fast enough for inference and safer for web servers
env = CheckersEnv()
models = {}
action_manager = ActionManager(device=device)
encoder = CheckersBoardEncoder()
agents_config = {"p1": "human", "p2": "human"}

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

# --- MODEL LOADER ---
def load_available_models():
    """Scans folders for .pth files"""
    # Resolve root directory (one level up from web_interface)
    root_dir = os.path.abspath(os.path.join(base_dir, '..'))
    project_root = os.path.abspath(os.path.join(root_dir, '..'))

    paths = [
        os.path.join(root_dir, "opponent_pool"),
        os.path.join(project_root, "opponent_pool"),
        os.path.join(root_dir, "checkpoints_gen11_decisive"),
        os.path.join(root_dir, "checkpoints_gen12_elite"),
        os.path.join(root_dir, "checkpoints_iron_league_v3"),
        os.path.join(project_root, "checkpoints"),
        os.path.join(root_dir, "gen12_elite_3500"),
        os.path.join(root_dir, "agents", "d3qn")
    ]
    model_files = {}
    for p in paths:
        if os.path.exists(p):
            for f in os.listdir(p):
                if f.endswith(".pth"):
                    name = f.replace(".pth", "")
                    full_path = os.path.join(p, f)
                    model_files[name] = full_path
    return model_files

def load_agent(name, path):
    """Loads a D3QN agent from disk"""
    global action_manager
    if action_manager is None:
        action_manager = ActionManager(device=device)

    try:
        model = D3QNModel(action_manager.action_dim, device)
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
            print(f"⚠️  Legacy model detected: {name}")
            model = LegacyDuelingDQN(action_manager.action_dim, device).to(device)
            model.load_state_dict(state_dict)
        
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None


# --- ROUTES ---

@app.route('/')
def index():
    available_agents = load_available_models()
    return render_template('index.html', agents=available_agents.keys())

@app.route('/start_game', methods=['POST'])
def start_game():
    global env, agents_config, models
    data = request.json
    
    agents_config['p1'] = data.get('p1', 'human')
    agents_config['p2'] = data.get('p2', 'human')
    
    # Reload environment
    env = CheckersEnv()
    state = env.reset()
    
    # Load requested models
    model_paths = load_available_models()
    models = {}
    
    for p in ['p1', 'p2']:
        name = agents_config[p]
        if name != 'human' and name != 'random':
            if name in model_paths:
                models[name] = load_agent(name, model_paths[name])
    
    return jsonify(get_board_state())

@app.route('/get_move', methods=['POST'])
def get_move():
    """Calculates a move for an AI agent"""
    global env
    
    current_player = env.current_player # 1 or -1
    agent_name = agents_config['p1'] if current_player == 1 else agents_config['p2']
    
    if agent_name == 'human':
        return jsonify({"error": "Waiting for human move"})
        
    legal_moves = env.get_legal_moves()
    if not legal_moves:
        return jsonify({"game_over": True, "winner": 0}) # Stalemate/Loss
    
    selected_move = None
    
    # AI LOGIC
    if agent_name == 'random':
        import random
        selected_move = random.choice(legal_moves)
    else:
        # Neural Network Move
        model = models.get(agent_name)
        if model:
            state_tensor = encoder.encode(env.board.get_state(), current_player).unsqueeze(0).to(device)
            with torch.no_grad():
                if hasattr(model, 'online'):
                    # Pass player_side if model supports it (Gen 12)
                    try:
                        q_values = model.online(state_tensor, player_side=1 if current_player == 1 else -1)
                    except TypeError:
                        q_values = model.online(state_tensor)
                else:
                    try:
                        q_values = model(state_tensor, player_side=1 if current_player == 1 else -1)
                    except TypeError:
                        q_values = model(state_tensor)
            
            # --- FIX FOR P2 PERSPECTIVE ---
            if current_player == -1:
                # 1. Flip legal moves to Canonical (P1) perspective
                normalized_moves, mapping = parse_legal_moves(legal_moves, action_manager)
                
                canonical_moves = [action_manager.flip_move(m) for m in normalized_moves]
                mask = action_manager.make_legal_action_mask(canonical_moves).to(device)
                
                # Map Canonical ID -> Absolute ID
                canonical_to_absolute = {}
                for i, cm in enumerate(canonical_moves):
                    cid = action_manager.get_action_id(cm)
                    if cid >= 0:
                        orig_move = normalized_moves[i]
                        aid = action_manager.get_action_id(orig_move)
                        canonical_to_absolute[cid] = aid
            else:
                # P1: Canonical = Absolute
                mask = action_manager.make_legal_action_mask(legal_moves).to(device)
                canonical_to_absolute = None
            
            # Mask illegal moves
            q_values[0, ~mask] = -float('inf')
            action_id = int(q_values.argmax().item())
            
            # If P2, action_id is Canonical. We need to map it back to Absolute.
            if current_player == -1 and canonical_to_absolute is not None:
                action_id = canonical_to_absolute.get(action_id, -1)
            
            # Translate ID back to Move
            move_struct = action_manager.get_move_from_id(action_id)
            
            # Match strict format
            for lm in legal_moves:
                if isinstance(lm, list):
                     if (tuple(lm[0][0]), tuple(lm[-1][1])) == move_struct: selected_move = lm; break
                elif len(lm) == 2:
                     if (tuple(lm[0]), tuple(lm[1])) == move_struct: selected_move = lm; break
            
            if not selected_move: selected_move = legal_moves[0] # Fallback
        else:
            # Fallback if model failed to load
            import random
            selected_move = random.choice(legal_moves)
            
    # Apply Move
    state, reward, done, info = env.step(selected_move)
    return jsonify(get_board_state())

@app.route('/human_move', methods=['POST'])
def human_move():
    """Processes a human click"""
    data = request.json
    move = data.get('move') # Format: [[r1,c1], [r2,c2]]
    
    # Convert list of lists to list of tuples
    if isinstance(move[0], list):
        # Check if legal
        legal = env.get_legal_moves()
        
        valid = False
        final_move = None
        
        # Convert JS coordinates to Python Tuples for comparison
        move_start = tuple(move[0])
        move_end = tuple(move[-1])
        
        for lm in legal:
            # Handle standard moves and multi-jumps
            if isinstance(lm, list):
                lm_start = lm[0][0]
                lm_end = lm[-1][1]
            else:
                lm_start = lm[0]
                lm_end = lm[1]
            
            if lm_start == move_start and lm_end == move_end:
                final_move = lm
                valid = True
                break
        
        if valid:
            env.step(final_move)
            return jsonify(get_board_state())
            
    return jsonify({"error": "Illegal Move"}), 400

def get_board_state():
    board = env.board.board.tolist() # 8x8 grid
    legal_moves = env.get_legal_moves()
    
    # Dynamic Game Over Check
    is_game_over = len(legal_moves) == 0
    winner = 0
    
    if is_game_over:
        winner = -1 if env.current_player == 1 else 1

    return {
        "board": board,
        "current_player": env.current_player,
        "legal_moves": serialize_moves(legal_moves),
        "game_over": is_game_over,
        "winner": winner
    }

def serialize_moves(moves):
    """Converts move tuples to list format for JSON"""
    serialized = []
    for m in moves:
        if isinstance(m, list): # Multi-jump: [((r,c), (r,c)), ...]
            # We just send start and end for UI simplicity
            start = m[0][0]
            end = m[-1][1]
            serialized.append([start, end])
        else: # Regular: ((r,c), (r,c))
            serialized.append([m[0], m[1]])
    return serialized

if __name__ == '__main__':
    app.run(debug=True, port=5000)