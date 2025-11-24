import argparse
import json
import os

from dama_env.env import DamaEnv
from agents.random_agent import RandomAgent
from agents.rl_agent import QLearningAgent, RLAgent
from agents.hybrid_agent import HybridAgent


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RL_MODEL_PATH = os.path.join(MODELS_DIR, "rl_q_table.pkl")
HYBRID_MODEL_PATH = os.path.join(MODELS_DIR, "hybrid_model.pkl")
CONFIG_PATH = os.path.join(BASE_DIR, "config_agents.json")


def load_table(path, fallback=None):
    if os.path.exists(path):
        with open(path, "rb") as f:
            import pickle
            return pickle.load(f)
    return fallback or {}


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {"red": "rl", "white": "random"}


def make_agent(agent_type, player, rl_q, hybrid_q):
    if agent_type == "random":
        return RandomAgent(name=f"Random {player}")
    if agent_type == "rl":
        return RLAgent(player=player, q_table=rl_q)
    if agent_type == "hybrid":
        return HybridAgent(player=player, q_table=hybrid_q or rl_q)
    raise ValueError(f"Unknown agent type: {agent_type}")


def play_game(red_type, white_type, render=False):
    env = DamaEnv()
    env.reset()

    rl_q = load_table(RL_MODEL_PATH, load_table(os.path.join(BASE_DIR, "q_table.pkl"), {}))
    hybrid_q = load_table(HYBRID_MODEL_PATH, rl_q)

    agent_red = make_agent(red_type, 1, rl_q, hybrid_q)
    agent_white = make_agent(white_type, -1, rl_q, hybrid_q)

    print(f"Red (Player 1) – {red_type.upper()} Agent")
    print(f"White (Player 2) – {white_type.upper()} Agent")

    done = False
    steps = 0
    info = {"winner": 0}

    while not done and steps < 400:
        if render:
            env.render()
            stats = env.piece_stats()
            print(f"Player to move: {env.current_player} | Red pieces: {stats[1]}, White pieces: {stats[-1]}")

        moving_player = env.current_player
        agent = agent_red if moving_player == 1 else agent_white
        action = agent.select_action(env)

        if action is None:
            done = True
            env.done = True
            info = {"winner": -moving_player}
            print("No legal moves.")
            break

        _, reward, done, info = env.step(action)

        if moving_player == 1 and hasattr(agent_red, "observe"):
            agent_red.observe(reward, env, done)
        if moving_player == -1 and hasattr(agent_white, "observe"):
            agent_white.observe(reward, env, done)

        if render:
            print(f"Action by {'Red' if moving_player == 1 else 'White'}: {action} | info: {info}")

        steps += 1

        if info.get("continue"):
            continue

    if render:
        env.render()
        print("Game over. Info:", info)


def parse_args():
    parser = argparse.ArgumentParser(description="Play one game with configurable agents.")
    parser.add_argument("--red", choices=["random", "rl", "hybrid"], help="Agent for Red (Player 1)")
    parser.add_argument("--white", choices=["random", "rl", "hybrid"], help="Agent for White (Player 2)")
    parser.add_argument("--no-render", action="store_true", help="Disable board rendering output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config()
    red = args.red or cfg.get("red", "rl")
    white = args.white or cfg.get("white", "random")
    play_game(red, white, render=not args.no_render)
