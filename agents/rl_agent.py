import random
import numpy as np


class QLearningAgent:
    def __init__(self, player, alpha=0.1, gamma=0.99, epsilon=0.2):
        """
        player: 1 or -1 (which side this agent plays)
        """
        self.player = player
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}  # (state_key, action_key) -> value
        self.last_state = None
        self.last_action_key = None
        self.pending_sequence = []  # remaining capture steps to honor chosen chain

    def _state_key(self, env):
        # State as a tuple + whose turn it is (from agent POV)
        board = env.board.get_state()
        flat = tuple(board.flatten())
        turn = env.current_player
        return (flat, turn)

    def _action_key(self, action):
        """
        Store Q per atomic step. Simple move => ((r1,c1),(r2,c2))
        Capture step or sequence => first step only.
        """
        if isinstance(action, (list, tuple)) and action and isinstance(action[0], (list, tuple)) and len(action[0]) == 3:
            step = action[0]
        elif isinstance(action, (list, tuple)) and len(action) == 3:
            step = action
        else:
            step = action
        if isinstance(step, (list, tuple)) and len(step) == 3:
            return tuple(tuple(p) for p in step)
        return (tuple(step[0]), tuple(step[1]))

    def _get_Q(self, state_key, action_key):
        return self.Q.get((state_key, action_key), 0.0)

    def select_action(self, env):
        # honor pending steps of a previously chosen capture chain
        if self.pending_sequence:
            action = self.pending_sequence.pop(0)
            self.last_state = self._state_key(env)
            self.last_action_key = self._action_key(action)
            return action

        moves = env.legal_moves()
        if not moves:
            self.last_state = None
            self.last_action_key = None
            return None

        state_key = self._state_key(env)

        # ε-greedy over available moves
        if random.random() < self.epsilon:
            action = random.choice(moves)
        else:
            qs = [self._get_Q(state_key, self._action_key(m)) for m in moves]
            max_q = max(qs)
            best_actions = [m for m, q in zip(moves, qs) if q == max_q]
            action = random.choice(best_actions)

        # If action is a multi-step capture sequence, queue remaining steps
        if isinstance(action, (list, tuple)) and action and isinstance(action[0], (list, tuple)) and len(action[0]) == 3:
            if len(action) > 1:
                self.pending_sequence = list(action[1:])
            action_to_play = action[0]
        elif isinstance(action, (list, tuple)) and len(action) == 3:
            action_to_play = action
        else:
            action_to_play = action

        self.last_state = state_key
        self.last_action_key = self._action_key(action_to_play)
        return action_to_play

    def observe(self, reward, next_env, done):
        if self.last_state is None or self.last_action_key is None:
            return

        old_key = self.last_state
        action_key = self.last_action_key
        old_q = self._get_Q(old_key, action_key)

        if done:
            target = reward
        else:
            next_key = self._state_key(next_env)
            moves = next_env.legal_moves()
            if moves:
                next_qs = [self._get_Q(next_key, self._action_key(a)) for a in moves]
                target = reward + self.gamma * max(next_qs)
            else:
                target = reward

        new_q = old_q + self.alpha * (target - old_q)
        self.Q[(old_key, action_key)] = new_q

        if done:
            self.last_state = None
            self.last_action_key = None
            self.pending_sequence = []


class RLAgent(QLearningAgent):
    """Inference-only agent that uses a precomputed Q-table."""

    def __init__(self, player, q_table, gamma=0.99, epsilon=0.0):
        super().__init__(player=player, alpha=0.0, gamma=gamma, epsilon=epsilon)
        self.Q = q_table

    def observe(self, reward, next_env, done):
        # Inference agent does not update Q-table
        return
