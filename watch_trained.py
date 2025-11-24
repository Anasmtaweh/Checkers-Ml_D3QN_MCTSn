from dama_env.env import DamaEnv
from agents.random_agent import RandomAgent
from agents.rl_agent import QLearningAgent
import pickle

def load_agent(path="q_table.pkl"):
    with open(path, "rb") as f:
        q = pickle.load(f)
    agent = QLearningAgent(player=1)
    agent.Q = q
    return agent

def watch_game():
    env = DamaEnv()
    env.reset()

    rl_agent = load_agent()
    random_agent = RandomAgent()

    done = False
    steps = 0

    while not done and steps < 200:
        env.render()
        print("Player to move:", env.current_player)
        input("Press Enter for next move...")

        if env.current_player == 1:
            action = rl_agent.select_action(env)
        else:
            action = random_agent.select_action(env)

        if action is None:
            next_state, reward, done, info = env.step(((0,0),(0,0)))
        else:
            next_state, reward, done, info = env.step(action)

        steps += 1

    env.render()
    print("Game Over:", info)

if __name__ == "__main__":
    watch_game()
