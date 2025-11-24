import argparse
import os
import pickle

from dama_env.env import DamaEnv
from agents.random_agent import RandomAgent
from agents.rl_agent import QLearningAgent, RLAgent
from agents.hybrid_agent import HybridAgent

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RL_MODEL_PATH = os.path.join(MODELS_DIR, "rl_q_table.pkl")
LEGACY_PATH = os.path.join(BASE_DIR, "q_table.pkl")


def ensure_models_dir():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)


def load_table():
    if os.path.exists(RL_MODEL_PATH):
        with open(RL_MODEL_PATH, "rb") as f:
            return pickle.load(f)
    if os.path.exists(LEGACY_PATH):
        with open(LEGACY_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def opponent_from_type(opponent, rl_q):
    if opponent == "random":
        return RandomAgent()
    if opponent == "hybrid":
        return HybridAgent(player=-1, q_table=rl_q)
    if opponent == "rl":
        # simple mirror RL as fixed policy; does not learn
        return RLAgent(player=-1, q_table=rl_q)
    raise ValueError(f"Unknown opponent type: {opponent}")


def train(opponent="random", episodes=2000):
    ensure_models_dir()
    init_q = load_table()
    rl_agent = QLearningAgent(player=1, alpha=0.1, gamma=0.99, epsilon=0.2)
    rl_agent.Q = init_q
    opp_agent = opponent_from_type(opponent, init_q)

    win_history = []

    for episode in range(1, episodes + 1):
        env = DamaEnv()
        env.reset()

        done = False
        steps = 0
        info = {"winner": 0}

        while not done and steps < 400:
            moving_player = env.current_player
            agent = rl_agent if moving_player == 1 else opp_agent
            action = agent.select_action(env)

            if action is None:
                done = True
                env.done = True
                info = {"winner": -moving_player}
                reward = -50 if moving_player == 1 else 50
                rl_agent.observe(reward, env, done)
                break

            _, reward, done, info = env.step(action)

            if moving_player == 1:
                rl_agent.observe(reward, env, done)

            steps += 1

            if info.get("continue"):
                continue

        winner = info["winner"]
        win_history.append(winner)

        if episode % 100 == 0:
            p1_wins = win_history.count(1)
            p2_wins = win_history.count(-1)
            draws = win_history.count(0)

            print(f"[Episode {episode}] "
                  f"P1 Wins: {p1_wins},  P2 Wins: {p2_wins}, Draws: {draws}")

            win_history = []  # reset stats window

    with open(RL_MODEL_PATH, "wb") as f:
        pickle.dump(rl_agent.Q, f)

    print(f"Training complete. Model saved to {RL_MODEL_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agent.")
    parser.add_argument("--opponent", choices=["random", "hybrid", "rl"], default="random", help="Opponent agent type")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(opponent=args.opponent, episodes=args.episodes)
