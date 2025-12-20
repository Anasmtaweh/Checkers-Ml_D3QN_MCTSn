import os
import sys
import torch
import numpy as np
from flask import Flask, render_template, jsonify, request

# Add parent directory to path so we can import your existing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from checkers_env.env import CheckersEnv
from training.common.action_manager import ActionManager
from training.common.board_encoder import CheckersBoardEncoder
from training.d3qn.model import D3QNModel

# Fix for TemplateNotFound: Explicitly define paths relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, 'templates')
static_dir = os.path.join(base_dir, 'static')

# Auto-detect flat structure if folders are missing
if not os.path.exists(template_dir):
    print(f"⚠️  Template dir '{template_dir}' not found. Using '{base_dir}'")
    template_dir = base_dir

if not os.path.exists(static_dir):
    print(f"⚠️  Static dir '{static_dir}' not found. Using '{base_dir}'")
    static_dir = base_dir

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# --- GLOBAL GAME STATE ---
device = "cpu"  # CPU is fast enough for inference and safer for web servers
env = CheckersEnv()
models = {}
action_manager = ActionManager(device=device)
encoder = CheckersBoardEncoder()
agents_config = {"p1": "human", "p2": "human"}

# --- MODEL LOADER ---
def load_available_models():
    """Scans folders for .pth files"""
    # Resolve root directory (one level up from web_interface)
    root_dir = os.path.abspath(os.path.join(base_dir, '..'))

    paths = [
        os.path.join(root_dir, "opponent_pool"),
        os.path.join(root_dir, "checkpoints_iron_league_v3"),
        os.path.join(root_dir, "checkpoints_iron_league"),
        os.path.join(root_dir, "checkpoints")
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
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_online' in checkpoint:
            model.online.load_state_dict(checkpoint['model_online'])
        else:
            model.online.load_state_dict(checkpoint)
        
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
                q_values = model.online(state_tensor)
            
            # Mask illegal moves
            mask = action_manager.make_legal_action_mask(legal_moves).to(device)
            q_values[0, ~mask] = -float('inf')
            action_id = int(q_values.argmax().item())
            
            # Translate ID back to Move
            move_struct = action_manager.get_move_from_id(action_id)
            
            # Match strict format
            for lm in legal_moves:
                if isinstance(lm, list):
                     if (tuple(lm[0][0]), tuple(lm[-1][1])) == move_struct: selected_move = lm; break
                elif len(lm) == 2:
                     if (tuple(lm[0]), tuple(lm[1])) == move_struct: selected_move = lm; break
            
            if not selected_move: selected_move = legal_moves[0] # Fallback
            
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