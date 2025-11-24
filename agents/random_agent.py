import random

class RandomAgent:
    def __init__(self, name="Random"):
        self.name = name

    def select_action(self, env):
        moves = env.legal_moves()
        if not moves:
            return None
        return random.choice(moves)

    def observe(self, *args, **kwargs):
        # Random agent doesn't learn
        pass
