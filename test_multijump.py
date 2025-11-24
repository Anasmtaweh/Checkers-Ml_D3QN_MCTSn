import numpy as np

from dama_env.env import DamaEnv

env = DamaEnv()

# custom board: force multi-jump scenario
env.board.board = np.array([
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,-1,0,0,0,0,0,0],
    [0,0,-1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
], dtype=int)

print("Initial:")
env.render()

moves = env.legal_moves()
print("\nLegal:", moves)

state, reward, done, info = env.step(moves[0])
env.render()
print("\nContinue?", info)

if info["continue"]:
    moves2 = env.legal_moves()
    print("\nNext capt:", moves2)
