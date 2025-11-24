import pickle
from dama_env.env import DamaEnv
from agents.random_agent import RandomAgent
from agents.hybrid_agent import HybridAgent

# load trained Q-table
with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

hybrid = HybridAgent(player=1, q_table=q_table)
random_agent = RandomAgent()

env = DamaEnv()
env.reset()

done = False
step = 0

while not done and step < 100:
    env.render()
    print("Player:", env.current_player)
    input("Enter to continue...")

    if env.current_player == 1:
        move = hybrid.choose(env)
    else:
        move = random_agent.select_action(env)

    if move is None:
        next_state, reward, done, info = env.step(((0,0),(0,0)))
    else:
        next_state, reward, done, info = env.step(move)

    step += 1

print("Game Over:", info)
env.render()
