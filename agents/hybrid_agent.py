import random
import hashlib
import numpy as np
from typing import Any

class HybridAgent:
    def __init__(self, player, q_table, gamma=0.99):
        self.player = player
        self.Q = q_table
        self.gamma = gamma

    def _is_capture_step(self, step):
        return isinstance(step, (list, tuple)) and len(step) == 3 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in step)

    def _first_step(self, move: Any):
        """Return the first atomic step of a move (sequence or simple)."""
        if isinstance(move, (list, tuple)) and move and self._is_capture_step(move[0]):
            return move[0]
        return move

    def _action_key(self, move: Any):
        step = self._first_step(move)
        if self._is_capture_step(step):
            return tuple(tuple(p) for p in step)
        return (tuple(step[0]), tuple(step[1]))

    def get_state_key(self, state):
        flat = tuple(state.flatten())
        return hashlib.sha1(str(flat).encode()).hexdigest()

    def q_value(self, state, action):
        key = self.get_state_key(state)
        action_key = self._action_key(action)
        return self.Q.get(key, {}).get(action_key, 0.0)

    def choose(self, env):
        legal = env.legal_moves()
        state = env.board.get_state()

        if not legal:
            return None

        # 1) Mandatory capture rule
        capture_moves = [m for m in legal if self._is_capture_step(self._first_step(m))]
        if capture_moves:
            return random.choice(capture_moves)

        # 2) Promotion moves
        promo_moves = []
        last_row = 7 if self.player == 1 else 0
        for m in legal:
            step = self._first_step(m)
            (_, _), (r2, _) = step
            if r2 == last_row:
                promo_moves.append(m)

        if promo_moves:
            return random.choice(promo_moves)

        # 3) Avoid-danger moves (simple version)
        safe_moves = []
        for move in legal:
            if not self.is_dangerous(env, move):
                safe_moves.append(move)

        if safe_moves:
            legal = safe_moves  # use safe-only list

        # 4) Pick best RL move
        best_score = -1e9
        best_move = None

        for move in legal:
            q = self.q_value(state, move)
            if q > best_score:
                best_score = q
                best_move = move

        if best_move is not None:
            return best_move

        # 5) Fallback: safe random or random move
        return random.choice(legal)

    def is_dangerous(self, env, move):
        """Check if opponent can capture us after this move."""
        step = self._first_step(move)
        (r1, c1), (r2, c2) = step

        # create a copy of env
        sim_board = np.copy(env.board.get_state())
        sim_board[r1, c1] = 0
        sim_board[r2, c2] = self.player

        # get opponent capture moves
        opp = -self.player
        from dama_env.rules import Rules
        opp_caps = Rules.get_legal_moves(sim_board, opp)
        for cap in opp_caps:
            first = self._first_step(cap)
            if self._is_capture_step(first):
                return True

        return False
