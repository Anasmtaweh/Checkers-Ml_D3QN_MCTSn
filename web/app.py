import os
import sys
import copy
import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flask import Flask, render_template, jsonify, request

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.game import CheckersEnv
from core.action_manager import ActionManager
from core.board_encoder import CheckersBoardEncoder
# Try importing D3QN, fallback if missing
try:
    from training.d3qn.model import D3QNModel
except ImportError:
    D3QNModel = None

from training.alpha_zero.network import AlphaZeroModel
import training.alpha_zero.mcts as mcts_module

base_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=base_dir, static_folder=base_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Web App running on: {device}")

# --- GLOBAL STATE ---
env = CheckersEnv()
models = {}
action_manager = ActionManager(device=device)
encoder = CheckersBoardEncoder()
agents_config = {"p1": "human", "p2": "human"}
game_lock = threading.Lock()
WEB_MCTS_SIMS = 1600
last_move_info = None

# --- LEGACY MODEL SUPPORT ---
class LegacyDuelingDQN(nn.Module):
    def __init__(self, action_dim, device):
        super(LegacyDuelingDQN, self).__init__()
        self.device = device
        # Legacy models were trained on 5 channels
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
    paths = [
        os.path.join(root_dir, "checkpoints", "alphazero"), 
        os.path.join(root_dir, "agents", "d3qn"),
        os.path.join(root_dir, "models") 
    ]
    model_files = {}
    for p in paths:
        if os.path.exists(p):
            for f in os.listdir(p):
                if f.endswith(".pth"): 
                    model_files[f.replace(".pth", "")] = os.path.join(p, f)
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
        "winner": int(env.winner),
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
            return jsonify(get_board_state())
        
        cp = env.current_player
        agent_name = agents_config['p1'] if cp == 1 else agents_config['p2']
        
        if agent_name == 'human': 
            return jsonify({"error": "Waiting for human"}), 200
        
        legal_moves = env.get_legal_moves()
        model = models.get(agent_name)
        selected_move = None
        
        if model:
            # === ALPHAZERO LOGIC (BEAST MODE UPDATED) ===
            if isinstance(model, AlphaZeroModel):
                mcts = mcts_module.MCTS(
                    model,
                    action_manager,
                    encoder,
                    c_puct=1.5,
                    num_simulations=WEB_MCTS_SIMS,
                    device=device,

                    # PURE AlphaZero inference (matches training)
                    dirichlet_alpha=0.0,
                    draw_value=0.0,
                    search_draw_bias=0.0
                )
                sim_env = copy.deepcopy(env)
                probs, root = mcts.get_action_prob(sim_env, temp=0.0, training=False)
                print(
                    f"[AI {agent_name}] "
                    f"V={root.get_greedy_value():.3f} | "
                    f"N={root.visits}"
                )
                best_action = int(np.argmax(probs))
                selected_move = mcts._get_move_from_action(best_action, legal_moves, player=cp)

            # === DQN/LEGACY LOGIC (Needs 5-Channel Compat) ===
            else:
                # 1. Encode 6 channels (Standard Gen2)
                state = encoder.encode(
                    env.board.get_state(), 
                    cp, 
                    force_move_from=env.force_capture_from
                ).unsqueeze(0).to(device)
                
                # 2. COMPATIBILITY FIX: SLICE TO 5 CHANNELS IF NEEDED
                # Check the first layer of the actual network being used
                network_to_check = model.online if hasattr(model, 'online') else model
                
                # If model starts with a Conv2d expecting 5 channels, strip the 6th
                if hasattr(network_to_check, 'conv1'):
                     if getattr(network_to_check.conv1, 'in_channels', 0) == 5:
                         if state.shape[1] == 6:
                             state = state[:, :5, :, :]
                
                # 3. Run Inference
                with torch.no_grad():
                    if hasattr(model, 'online'):
                        q_values = model.online(state)
                    elif hasattr(model, 'get_q_values'):
                        q_values = model.get_q_values(state)
                    else:
                        q_values = model(state)

                # 4. Masking & Selection
                mask = action_manager.make_legal_action_mask(legal_moves, player=cp).to(device)
                q_values[0, ~mask] = -float('inf')
                
                action_id = int(torch.argmax(q_values).item())
                
                # Map back to move
                move_pair = action_manager.get_move_from_id(action_id)
                if cp == -1: 
                    move_pair = action_manager.flip_move(move_pair)
                
                for m in legal_moves:
                    if action_manager._extract_start_landing(m) == move_pair:
                        selected_move = m
                        break
        else:
            import random
            selected_move = random.choice(legal_moves)
        
        if selected_move is None:
            print("⚠️ Agent failed to select move. Picking random.")
            import random
            selected_move = legal_moves[0]

        if selected_move:
            # selected_move is usually [[r1,c1], [r2,c2]]
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

        for m in env.get_legal_moves():
            m_start = tuple(m[0])
            m_landing = tuple(m[1])
            if m_start == start and m_landing == landing:
                last_move_info = {'start': list(start), 'end': list(landing)}
                env.step(m)
                return jsonify(get_board_state())

        return jsonify({"error": "Illegal Move"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)