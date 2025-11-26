import os
import random
import hashlib
import pickle
import numpy as np
from typing import Any, Optional

DEFAULT_HYBRID_QTABLE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "q_tables", "q_table_hybrid.pkl")


class HybridAgent:
    def __init__(self, player, q_table=None, gamma=0.99, alpha=0.1, epsilon=0.2, q_table_path: Optional[str] = DEFAULT_HYBRID_QTABLE):
        self.player = player
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.q_table_path = q_table_path
        self.Q = q_table if q_table is not None else self._load_table()
        self.pending_sequence = []

    def _load_table(self):
        if self.q_table_path and os.path.exists(self.q_table_path):
            with open(self.q_table_path, "rb") as f:
                return pickle.load(f)
        return {}

    def save(self):
        if not self.q_table_path:
            return
        os.makedirs(os.path.dirname(self.q_table_path), exist_ok=True)
        with open(self.q_table_path, "wb") as f:
            pickle.dump(self.Q, f)

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
        return self.Q.get((key, action_key), 0.0)

    def _max_next_q(self, next_state, legal):
        if not legal:
            return 0.0
        key = self.get_state_key(next_state)
        values = [self.Q.get((key, self._action_key(a)), 0.0) for a in legal]
        return max(values) if values else 0.0

    def update_q(self, state, action, reward, next_state, next_legal, done):
        s_key = self.get_state_key(state)
        a_key = self._action_key(action)
        old_q = self.Q.get((s_key, a_key), 0.0)
        if done:
            target = reward
        else:
            target = reward + self.gamma * self._max_next_q(next_state, next_legal)
        self.Q[(s_key, a_key)] = old_q + self.alpha * (target - old_q)

    def choose(self, env, training: bool = False):
        # honor pending sequence during capture chains
        if self.pending_sequence:
            return self.pending_sequence.pop(0)

        legal = env.legal_moves()
        state = env.board.get_state()

        if not legal:
            return None

        # 1) Mandatory capture rule via env ensures captures listed; prefer capture immediately
        capture_moves = [m for m in legal if self._is_capture_step(self._first_step(m))]
        if capture_moves:
            action = random.choice(capture_moves)
        else:
            # epsilon-greedy over Q for non-captures (or captures if none flagged)
            if training and random.random() < self.epsilon:
                action = random.choice(legal)
            else:
                qs = [self.q_value(state, m) for m in legal]
                max_q = max(qs)
                best = [m for m, q in zip(legal, qs) if q == max_q]
                action = random.choice(best)

        # queue additional steps if this is a multi-step capture
        if isinstance(action, (list, tuple)) and action and self._is_capture_step(action[0]):
            if len(action) > 1:
                self.pending_sequence = list(action[1:])
            return action[0]
        return action

    def observe(self, state, action, reward, next_state, next_legal, done):
        self.update_q(state, action, reward, next_state, next_legal, done)
        if done:
            self.save()

    def is_dangerous(self, env, move):
        """Check if opponent can capture us after this move."""
        step = self._first_step(move)
        (r1, c1), (r2, c2) = step

        sim_board = np.copy(env.board.get_state())
        sim_board[r1, c1] = 0
        sim_board[r2, c2] = self.player

        opp = -self.player
        from dama_env.rules import Rules
        opp_caps = Rules.get_legal_moves(sim_board, opp)
        for cap in opp_caps:
            first = self._first_step(cap)
            if self._is_capture_step(first):
                return True
        return False
