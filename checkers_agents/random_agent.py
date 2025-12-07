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

