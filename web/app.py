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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.game import CheckersEnv
from core.action_manager import ActionManager
from core.board_encoder import CheckersBoardEncoder
from core.move_parser import parse_legal_moves
from training.d3qn.model import D3QNModel
from training.alpha_zero.network import AlphaZeroModel
from training.alpha_zero.mcts import MCTS

base_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=base_dir, static_folder=base_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Web App running on: {device}")

env = CheckersEnv()
models = {}
action_manager = ActionManager(device=device)
encoder = CheckersBoardEncoder()
agents_config = {"p1": "human", "p2": "human"}
game_lock = threading.Lock()
WEB_MCTS_SIMS = 400 

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

def load_available_models():
    root_dir = os.path.abspath(os.path.join(base_dir, '..'))
    paths = [os.path.join(root_dir, "checkpoints", "alphazero"), os.path.join(root_dir, "agents", "d3qn")]
    model_files = {}
    for p in paths:
        if os.path.exists(p):
            for f in os.listdir(p):
                if f.endswith(".pth"): model_files[f.replace(".pth", "")] = os.path.join(p, f)
    return model_files

def load_agent(name, path):
    print(f"Loading agent: {name}")
    try:
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model = AlphaZeroModel(action_dim=action_manager.action_dim, device=device)
            model.network.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model
        
        state_dict = checkpoint.get('model_online', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        try:
            model = D3QNModel(action_manager.action_dim, device)
            model.online.load_state_dict(state_dict)
            model.eval()
            return model
        except:
            model = LegacyDuelingDQN(action_manager.action_dim, device).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            return model
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_board_state():
    return {
        "board": env.board.board.tolist(),
        "current_player": env.current_player,
        "legal_moves": serialize_moves(env.get_legal_moves()),
        "game_over": len(env.get_legal_moves()) == 0,
        "winner": -env.current_player if len(env.get_legal_moves()) == 0 else 0
    }

def serialize_moves(moves):
    return [[m[0][0], m[-1][1]] if isinstance(m, list) else [m[0], m[1]] for m in moves]

@app.route('/')
def index(): return render_template('index.html', agents=list(load_available_models().keys()))

@app.route('/start_game', methods=['POST'])
def start_game():
    with game_lock:
        data = request.json
        agents_config['p1'] = data.get('p1', 'human')
        agents_config['p2'] = data.get('p2', 'human')
        env.reset()
        models.clear()
        paths = load_available_models()
        for p in ['p1', 'p2']:
            name = agents_config[p]
            if name != 'human' and name in paths: models[name] = load_agent(name, paths[name])
        return jsonify(get_board_state())

@app.route('/get_move', methods=['POST'])
def get_move():
    with game_lock:
        if not env.get_legal_moves(): return jsonify(get_board_state())
        
        cp = env.current_player
        name = agents_config['p1'] if cp == 1 else agents_config['p2']
        if name == 'human': return jsonify({"error": "Waiting"})
        
        legal = env.get_legal_moves()
        model = models.get(name)
        selected = None
        move_struct = None
        
        if model:
            if isinstance(model, AlphaZeroModel):
                mcts = MCTS(model, action_manager, encoder, c_puct=1.5, num_simulations=WEB_MCTS_SIMS, device=device, dirichlet_alpha=0.0)
                
                # --- FIX: Create a VIEW of the environment for the AI ---
                # The AI (AlphaZero) is trained to always play as "Player 1" (moving Up).
                # If the current player is "Player -1" (Black), we must present the board 
                # as if they are Red. MCTS usually handles logic, but let's ensure 
                # legal moves are correctly masked.
                
                # We pass the real env. The MCTS class *should* handle the flipping internally 
                # if implemented correctly with the encoder. 
                # BUT, since we are seeing errors, we will force the flip logic here.
                
                sim_env = copy.deepcopy(env)
                
                # Run MCTS
                probs, root_node = mcts.get_action_prob(sim_env, temp=0.25, training=False)
                
                # --- DEBUG LOGS ---
                print(f"\n[DEBUG] AI Thinking (Player {cp})...")
                print(f"  Root Value (My Winning Chance): {root_node.get_greedy_value():.4f}")
                
                # ... (Debug printing logic) ...
                
                aid = int(np.argmax(probs))
                move_struct = action_manager.get_move_from_id(aid)
                
                # CRITICAL FIX: If we are Player -1, the AI output a "Canonical" move (Red perspective).
                # We must FLIP it back to Real World coordinates to execute it.
                if cp == -1: 
                    move_struct = action_manager.flip_move(move_struct)
            else:
                # D3QN Logic (Keep existing)
                state = encoder.encode(env.board.get_state(), cp).unsqueeze(0).to(device)
                with torch.no_grad():
                    q = model.online(state) if hasattr(model, 'online') else model(state)
                
                if cp == -1:
                    norm, _ = parse_legal_moves(legal, action_manager)
                    can = [action_manager.flip_move(m) for m in norm]
                    mask = action_manager.make_legal_action_mask(can).to(device)
                    can_to_abs = {action_manager.get_action_id(c): action_manager.get_action_id(n) for c, n in zip(can, norm)}
                else:
                    mask = action_manager.make_legal_action_mask(legal).to(device)
                    can_to_abs = None

                q[0, ~mask] = -float('inf')
                probs = F.softmax(q/0.1, dim=1)
                aid = int(torch.multinomial(probs, 1).item())
                
                if cp == -1 and can_to_abs: aid = can_to_abs.get(aid, -1)
                move_struct = action_manager.get_move_from_id(aid)

            for m in legal:
                start, end = (m[0][0], m[0][1]) if isinstance(m, list) else (m[0], m[1])
                if (tuple(start), tuple(end)) == move_struct:
                    selected = m
                    break
        else:
            import random
            selected = random.choice(legal)
        
        if not selected: 
            print(f"⚠️ Warning: Move {move_struct} illegal. Picking random fallback.")
            selected = legal[0]
            
        env.step(selected)
        print(f"Executed: {selected}")
        time.sleep(1.0)
        return jsonify(get_board_state())

@app.route('/human_move', methods=['POST'])
def human_move():
    with game_lock:
        move_data = request.json.get('move')
        if not move_data: return jsonify({"error": "No move"}), 400
        start, end = tuple(move_data[0]), tuple(move_data[-1])
        
        for m in env.get_legal_moves():
            m_start = m[0][0] if isinstance(m, list) else m[0]
            m_end = m[-1][1] if isinstance(m, list) else m[1]
            if m_start == start and m_end == end:
                env.step(m)
                return jsonify(get_board_state())
        return jsonify({"error": "Illegal"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)