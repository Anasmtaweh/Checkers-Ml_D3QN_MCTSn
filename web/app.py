import os
import sys
import pickle
from typing import Optional, Union, Dict

from flask import Flask, jsonify, request, render_template

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from checkers_env.env import CheckersEnv
from checkers_agents.random_agent import CheckersRandomAgent
from checkers_agents.ddqn_agent import DDQNAgent
from checkers_agents.mcts_agent import MCTSAgent

app = Flask(__name__, template_folder="templates", static_folder="static")

# Model paths
MODELS_DIR = os.path.join(base_dir, "models")
RL_MODEL_PATH = os.path.join(MODELS_DIR, "rl_q_table.pkl")
HYBRID_MODEL_PATH = os.path.join(MODELS_DIR, "hybrid_model.pkl")


def _load_table(paths):
    for p in paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    return {}


# Load tables (fallback to legacy q_table.pkl if models directory missing)
q_table = _load_table([RL_MODEL_PATH, os.path.join(base_dir, "q_table.pkl")])
hybrid_table = _load_table([HYBRID_MODEL_PATH, RL_MODEL_PATH, os.path.join(base_dir, "q_table.pkl")])

AgentType = Union[CheckersRandomAgent, DDQNAgent, MCTSAgent]
env: Optional[CheckersEnv] = None
agent1: Optional[AgentType] = None
agent2: Optional[AgentType] = None
agent_labels: Dict[str, str] = {"red": "random", "white": "random"}


def make_agent(agent_type: str, player: int):
    if agent_type == "random":
        return CheckersRandomAgent(name=f"Random {player}")
    if agent_type == "rl":
        return DDQNAgent(env=env)
    if agent_type == "mcts":
        return MCTSAgent(env=env)
    raise ValueError(f"Unknown agent type: {agent_type}")


def score_payload():
    if env is None:
        return {}
    stats = env.piece_stats()
    return {
        "agents": {
            "red": agent_labels.get("red", "unknown"),
            "white": agent_labels.get("white", "unknown")
        },
        "pieces": {
            "red": stats[1],
            "white": stats[-1]
        },
        "current_player": env.current_player
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start_game", methods=["POST"])
def start_game():
    global env, agent1, agent2

    data = request.get_json(silent=True) or {}
    mode = data.get("mode")
    red = data.get("red")
    white = data.get("white")

    if mode == "random_vs_rl":
        red, white = "random", "rl"
    elif mode == "rl_vs_hybrid":
        red, white = "rl", "mcts"
    elif mode == "hybrid_vs_hybrid":
        red, white = "mcts", "mcts"

    red = red or "rl"
    white = white or "random"

    env = CheckersEnv()
    env.reset()

    agent1 = make_agent(red, player=1)
    agent2 = make_agent(white, player=-1)
    agent_labels["red"] = red
    agent_labels["white"] = white

    return jsonify({"status": "ok", "score": score_payload()})


@app.route("/play_step", methods=["POST"])
def play_step():
    global env, agent1, agent2

    if env is None:
        return jsonify({"error": "Game not started"}), 400

    agent = agent1 if env.current_player == 1 else agent2
    if agent is None:
        return jsonify({"error": "Agents not initialized"}), 400

    if isinstance(agent, (CheckersRandomAgent, DDQNAgent, MCTSAgent)):
        move = agent.select_action(env)
    else:
        return jsonify({"error": "Unknown agent type"}), 400

    if move is None:
        # No legal moves available
        state = env.board.get_state()
        return jsonify({
            "board": state.tolist(),
            "current_player": env.current_player,
            "done": True,
            "move": None,
            "info": {"error": "No legal moves"},
            "score": score_payload()
        })

    state, reward, done, info = env.step(move)

    return jsonify({
        "board": state.tolist(),
        "current_player": env.current_player,
        "done": done,
        "info": info,
        "move": move,
        "score": score_payload()
    })


@app.route("/get_state")
def get_state():
    global env
    if env is None:
        return jsonify({"board": [], "current_player": 0, "score": {}})
    return jsonify({
        "board": env.board.get_state().tolist(),
        "current_player": env.current_player,
        "score": score_payload()
    })


if __name__ == "__main__":
    app.run(debug=True)
