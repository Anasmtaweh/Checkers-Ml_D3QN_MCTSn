from dama_env.env import DamaEnv

env = DamaEnv()
env.reset()

print("Initial:")
env.render()

moves = env.legal_moves()
print("\nLegal moves available:", moves)

# Let's try executing ALL moves to ensure environment doesn't break
for m in moves:
    print("\nTrying move:", m)
    envTest = DamaEnv()
    envTest.reset()
    envTest.step(m)
    envTest.render()

