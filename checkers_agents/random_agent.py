import random

class CheckersRandomAgent:
    """
    Simple random agent for evaluation.
    Expected API: select_action(env)
    """

    def select_action(self, env):
        legal_moves = env.get_legal_moves() if hasattr(env, "get_legal_moves") else env.legal_moves()
        if not legal_moves:
            return None
        return random.choice(legal_moves)


class RandomAgent:
    """
    Random agent compatible with D3QNTrainer opponent interface.
    """
    def __init__(self, device=None):
        self.device = device

    def select_action(self, board, player, legal_moves, epsilon=0.0, info=None):
        if not legal_moves:
            return None, None
        return random.choice(legal_moves), 0
