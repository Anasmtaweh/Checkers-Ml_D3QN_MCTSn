import os
import sys
import time

# Force CPU to avoid CUDA initialization errors on web server
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import copy
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flask import Flask, render_template, jsonify, request

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

# Add Root (ML_Gen2)
sys.path.append(ROOT_DIR)
# Add MCTS Workspace
sys.path.append(os.path.join(ROOT_DIR, 'mcts_workspace'))
# Add D3QN Workspace
sys.path.append(os.path.join(ROOT_DIR, 'd3qn_workspace'))

from mcts_workspace.core.game import CheckersEnv
from mcts_workspace.core.action_manager import ActionManager
from mcts_workspace.core.board_encoder import CheckersBoardEncoder
from mcts_workspace.training.alpha_zero.network import AlphaZeroModel
import mcts_workspace.training.alpha_zero.mcts as mcts_module

# --- OPTIONAL IMPORTS (with Pylance suppression) ---
D3QNModel = None
try:
    # Try importing assuming d3qn_workspace is in sys.path
    from training.d3qn.model import D3QNModel # type: ignore
except ImportError:
    try:
        # Try importing via full namespace if defined differently
        from d3qn_workspace.training.d3qn.model import D3QNModel # type: ignore
    except ImportError:
        print("⚠️ D3QN Model class not found. DQN agents will fail to load.")
        D3QNModel = None

base_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=base_dir, static_folder=base_dir)

# --- 1. CONFIGURATION: CPU + 1600 SIMS ---
device = "cpu" 
print(f"🔥 Web App running on: {device}")

WEB_MCTS_SIMS = 1600
print(f"⚡ MCTS Simulations set to: {WEB_MCTS_SIMS} (Full Strength)")

# --- GLOBAL STATE ---
env = CheckersEnv()
models = {}
action_manager = ActionManager(device=device)
encoder = CheckersBoardEncoder()
agents_config = {"p1": "human", "p2": "human"}
game_lock = threading.Lock()
last_move_info = None

# --- LEGACY MODEL SUPPORT ---
class LegacyDuelingDQN(nn.Module):
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

    def get_q_values(self, state, player_side=1): return self.forward(state)

# --- HELPERS ---
def load_available_models():
    root_dir = os.path.abspath(os.path.join(base_dir, '..'))
    model_files = {}

    # 1. AlphaZero Checkpoints
    az_path = os.path.join(root_dir, "mcts_workspace", "checkpoints", "alphazero")
    if os.path.exists(az_path):
        az_files = []
        for f in os.listdir(az_path):
            if f.endswith(".pth") and "iter_" in f:
                try:
                    iter_num = int(f.split("iter_")[-1].replace(".pth", ""))
                    if 200 <= iter_num <= 229:
                        az_files.append((iter_num, f))
                except ValueError:
                    pass
        
        for _, f in sorted(az_files):
            model_files[f.replace(".pth", "")] = os.path.join(az_path, f)

    # 2. D3QN Checkpoints
    d3qn_path = os.path.join(root_dir, "agents", "d3qn")
    if os.path.exists(d3qn_path):
        for f in os.listdir(d3qn_path):
            if f.endswith(".pth"):
                model_files[f.replace(".pth", "")] = os.path.join(d3qn_path, f)

    return model_files

def load_agent(name, path):
    print(f"Loading agent: {name} from {path}")
    try:
        checkpoint = torch.load(path, map_location=device)
        
        # Detect AlphaZero
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("  -> Detected AlphaZero Checkpoint")
            model = AlphaZeroModel(action_dim=action_manager.action_dim, device=device)
            model.network.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model
        
        # Fallback to DQN
        state_dict = checkpoint.get('model_online', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        
        try:
            if D3QNModel:
                model = D3QNModel(action_manager.action_dim, device)
                model.online.load_state_dict(state_dict)
                model.eval()
                return model
            else:
                raise ImportError("D3QN class missing")
        except:
            print("  -> Fallback to LegacyDuelingDQN")
            model = LegacyDuelingDQN(action_manager.action_dim, device).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            return model

    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
        return None

def serialize_moves(moves):
    out = []
    for m in moves:
        if isinstance(m, (list, tuple)) and len(m) == 3:
            out.append([[m[0][0], m[0][1]], [m[1][0], m[1][1]], [m[2][0], m[2][1]]])
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            out.append([[m[0][0], m[0][1]], [m[1][0], m[1][1]]])
    return out

def get_board_state():
    return {
        "board": env.board.board.tolist(),
        "current_player": env.current_player,
        "legal_moves": serialize_moves(env.get_legal_moves()),
        "game_over": bool(env.done),
        "winner": int(env.winner) if env.winner is not None else None,
        "last_move": last_move_info
    }

# --- ROUTES ---

@app.route('/')
def index(): 
    return render_template('index.html', agents=list(load_available_models().keys()))

@app.route('/start_game', methods=['POST'])
def start_game():
    with game_lock:
        global last_move_info
        last_move_info = None
        data = request.json
        agents_config['p1'] = data.get('p1', 'human')
        agents_config['p2'] = data.get('p2', 'human')
        env.reset()
        models.clear()
        
        avail = load_available_models()
        for p in ['p1', 'p2']:
            name = agents_config[p]
            if name != 'human' and name in avail: 
                models[name] = load_agent(name, avail[name])
                
        return jsonify(get_board_state())

@app.route('/get_move', methods=['POST'])
def get_move():
    with game_lock:
        global last_move_info
        if env.done or not env.get_legal_moves(): 
            if env.done:
                print(f"🏁 GAME OVER. Winner: {env.winner}")
            return jsonify(get_board_state())
        
        cp = env.current_player
        agent_name = agents_config['p1'] if cp == 1 else agents_config['p2']
        
        if agent_name == 'human': 
            return jsonify({"error": "Waiting for human"}), 200
        
        legal_moves = env.get_legal_moves()
        model = models.get(agent_name)
        selected_move = None
        
        if model:
            # === ALPHAZERO LOGIC (GOD MODE: NO NOISE, MAX IQ) ===
            if isinstance(model, AlphaZeroModel):
                mcts = mcts_module.MCTS(
                    model,
                    action_manager,
                    encoder,
                    c_puct=1.5,
                    num_simulations=5000,  # Ensure this is 1600
                    device=device,
                    
                    # 🛑 CRITICAL SETTINGS FOR DEMO 🛑
                    dirichlet_epsilon=0.0, # MUST BE 0.0. No random noise.
                    dirichlet_alpha=0.0,   # Disable Dirichlet entirely.
                    draw_value=0.0,        # Play Pure Chess (Checkers).
                    search_draw_bias=0.0   # No artificial bias.
                )
                
                sim_env = copy.deepcopy(env)
                
                start_time = time.time()
                
                # temp=0.0 -> Pick the move with highest visits (Deterministic)
                probs, root = mcts.get_action_prob(sim_env, temp=0.0, training=False) 
                
                duration = time.time() - start_time
                
                print(f"[AI {agent_name}] V={root.get_greedy_value():.3f} | N={root.visits} | T={duration:.2f}s")
                best_action = int(np.argmax(probs))
                selected_move = mcts._get_move_from_action(best_action, legal_moves, player=cp)

            # === DQN LOGIC ===
            else:
                state = encoder.encode(
                    env.board.get_state(), 
                    cp, 
                    force_move_from=env.force_capture_from
                ).unsqueeze(0).to(device)
                
                # Compat Fix: Slice 6ch -> 5ch if model requires it
                network_to_check = model.online if hasattr(model, 'online') else model
                if hasattr(network_to_check, 'conv1'):
                     if getattr(network_to_check.conv1, 'in_channels', 0) == 5:
                         if state.shape[1] == 6:
                             state = state[:, :5, :, :]
                
                with torch.no_grad():
                    if hasattr(model, 'online'):
                        q_values = model.online(state)
                    elif hasattr(model, 'get_q_values'):
                        q_values = model.get_q_values(state)
                    else:
                        q_values = model(state)

                mask = action_manager.make_legal_action_mask(legal_moves, player=cp).to(device)
                q_values[0, ~mask] = -float('inf')
                action_id = int(torch.argmax(q_values).item())
                
                move_pair = action_manager.get_move_from_id(action_id)
                if cp == -1: move_pair = action_manager.flip_move(move_pair)
                
                for m in legal_moves:
                    if action_manager._extract_start_landing(m) == move_pair:
                        selected_move = m
                        break
        else:
            import random
            selected_move = random.choice(legal_moves)
        
        if selected_move is None:
            import random
            selected_move = legal_moves[0]

        start = [int(selected_move[0][0]), int(selected_move[0][1])]
        end = [int(selected_move[1][0]), int(selected_move[1][1])]
        last_move_info = {'start': start, 'end': end}

        env.step(selected_move)
        return jsonify(get_board_state())

@app.route('/human_move', methods=['POST'])
def human_move():
    with game_lock:
        global last_move_info
        if env.done: return jsonify(get_board_state())
        
        cp = env.current_player
        agent_name = agents_config['p1'] if cp == 1 else agents_config['p2']
        if agent_name != 'human':
            return jsonify({"error": "Not human turn"}), 400
            
        move_data = request.json.get('move')
        if not move_data: return jsonify({"error": "No move"}), 400
        
        start = tuple(move_data[0])
        landing = tuple(move_data[1])

        legal_moves = env.get_legal_moves()
        for m in legal_moves:
            if tuple(m[0]) == start and tuple(m[1]) == landing:
                last_move_info = {'start': list(start), 'end': list(landing)}
                env.step(m)
                return jsonify(get_board_state())

        print(f"⚠️ REJECTED MOVE: {start} -> {landing}")
        print(f"   Must be one of: {[ (m[0], m[1]) for m in legal_moves ]}")
        return jsonify({"error": "Illegal Move"}), 400

if __name__ == '__main__':
    # usage_reloader=False stops it from watching your files
    app.run(debug=True, use_reloader=False, port=5000, threaded=True)