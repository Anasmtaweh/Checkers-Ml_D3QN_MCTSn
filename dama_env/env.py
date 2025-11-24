from .board import Board
from .rules import Rules
import numpy as np
from typing import List, Tuple, Union

Move = Tuple[Tuple[int, int], Tuple[int, int]]  # simple move
CaptureStep = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]


class DamaEnv:
    def __init__(self):
        self.board = Board()
        self.current_player = 1  # player to move
        self.done = False
        self.force_capture_from = None  # position that must continue a chain

    def reset(self):
        self.board.reset()
        self.current_player = 1
        self.done = False
        self.force_capture_from = None
        return self.board.get_state()

    def _count_pieces(self):
        b = self.board.get_state()
        p1 = np.sum((b == 1) | (b == 2))
        p2 = np.sum((b == -1) | (b == -2))
        return p1, p2

    def piece_stats(self):
        """Return piece and king counts for both players."""
        b = self.board.get_state()
        stats = {
            1: {"pieces": int(np.sum(b == 1)), "kings": int(np.sum(b == 2))},
            -1: {"pieces": int(np.sum(b == -1)), "kings": int(np.sum(b == -2))}
        }
        return stats

    def _check_game_over(self):
        """Returns (done, winner) where winner is 1, -1, or 0 (draw/none)."""
        p1, p2 = self._count_pieces()
        if p1 == 0 and p2 == 0:
            return True, 0
        if p1 == 0:
            return True, -1
        if p2 == 0:
            return True, 1

        # if side to move has no legal moves → they lose
        moves = Rules.get_legal_moves(self.board.get_state(), self.current_player)
        if len(moves) == 0:
            winner = -self.current_player
            return True, winner

        return False, 0

    def _is_capture_step(self, action) -> bool:
        return (
            isinstance(action, (list, tuple))
            and len(action) == 3
            and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in action)
        )

    def _normalize_action(self, action) -> Union[List[CaptureStep], Move]:
        """
        Convert incoming action into either a list of capture steps or a simple move.
        Capture action can be a single step or a full sequence (list of steps).
        """
        if self._is_capture_step(action):
            return [tuple(tuple(p) for p in action)]  # single capture step
        if isinstance(action, (list, tuple)) and action and self._is_capture_step(action[0]):
            return [tuple(tuple(p) for p in step) for step in action]  # full sequence
        # simple move ((r1,c1), (r2,c2))
        return (tuple(action[0]), tuple(action[1]))

    def _apply_capture_sequence(self, steps: List[CaptureStep], player: int):
        """Apply one or more capture steps. Returns (last_pos, total_reward, continue_flag)."""
        total_reward = 0.0
        last_pos = None

        for start, landing, jumped in steps:
            sr, sc = start
            lr, lc = landing
            jr, jc = jumped

            self.board.move_piece(sr, sc, lr, lc)
            self.board.board[jr, jc] = 0

            last_pos = (lr, lc)
            total_reward += 10  # per captured piece
            total_reward -= 1   # per move penalty

        # check if another capture is possible from last_pos
        more_captures = Rules.capture_sequences(self.board.get_state(), player, start_pos=last_pos)
        if more_captures:
            self.force_capture_from = last_pos
            return last_pos, total_reward, True

        self.force_capture_from = None
        return last_pos, total_reward, False

    def step(self, action):
        if self.done:
            return self.board.get_state(), 0.0, True, {"winner": 0}

        normalized = self._normalize_action(action)
        player = self.current_player
        info = {"winner": 0}

        # Capture path
        if isinstance(normalized, list):
            last_pos, reward, must_continue = self._apply_capture_sequence(normalized, player)

            if must_continue:
                # Same player continues the chain; game is not over yet
                self.done = False
                info["continue"] = True
                info["from"] = last_pos
                return self.board.get_state(), reward, False, info

            # no more captures; switch turn after chain ends
            self.current_player *= -1
        else:
            # simple move
            (r1, c1), (r2, c2) = normalized
            self.board.move_piece(r1, c1, r2, c2)
            reward = -1  # per move penalty
            self.force_capture_from = None
            self.current_player *= -1

        done, winner = self._check_game_over()
        self.done = done
        info["winner"] = winner

        if done:
            if winner == player:
                reward += 50
            elif winner == -player:
                reward -= 50

        return self.board.get_state(), reward, done, info

    def legal_moves(self):
        if self.done:
            return []
        return Rules.get_legal_moves(self.board.get_state(), self.current_player, self.force_capture_from)

    def render(self):
        self.board.print_board()
