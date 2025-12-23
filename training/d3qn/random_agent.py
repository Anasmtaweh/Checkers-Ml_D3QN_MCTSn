"""Simple random agent for checkers."""
import random


class RandomAgent:
    """Agent that plays random valid moves."""
    
    def select_action(self, env):
        """Select a random legal move."""
        legal_moves = env.get_legal_moves()
        if legal_moves:
            return random.choice(legal_moves)
        return None
