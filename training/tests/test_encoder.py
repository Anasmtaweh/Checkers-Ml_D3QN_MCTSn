from checkers_env.env import CheckersEnv
from training.common.board_encoder import CheckersBoardEncoder

env = CheckersEnv()
encoder = CheckersBoardEncoder()

state = env.reset()
encoded = encoder.encode(state)

print("Encoded shape:", encoded.shape)
print("Red men plane sum:", encoded[0].sum())
print("Red turn plane unique values:", encoded[6].unique())
