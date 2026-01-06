from .board import CheckersBoard
from .rules import CheckersRules
import numpy as np
from typing import List, Tuple, Any, Optional, Dict

Move = Tuple[Tuple[int, int], Tuple[int, int]]  # simple move
CaptureStep = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]  # (start, landing, jumped)


class CheckersEnv:
    """
    Environment wrapper around CheckersBoard + CheckersRules with
    reward shaping for:
      - winning / losing
      - captures (incl. multi-jump bonus)
      - king promotion
      - light positional heuristics (center control, avoiding obvious blunders)
    """

    def __init__(self, max_moves: int = 200, no_progress_limit: int = 80):
        self.board = CheckersBoard()
        self.current_player = 1  # player to move
        self.done = False
        self.winner = 0
        # When a capture chain is in progress, this is the square that must continue
        self.force_capture_from: Optional[Tuple[int, int]] = None
        self.move_count = 0
        self.max_moves = max_moves
        self.no_progress = 0
        self.no_progress_limit = no_progress_limit

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def reset(self):
        """Reset to starting position and return the board array."""
        self.board.reset()
        self.current_player = 1
        self.done = False
        self.winner = 0
        self.force_capture_from = None
        self.move_count = 0
        self.no_progress = 0
        return self.board.get_state()

    def _piece_and_king_counts_for(self, board: np.ndarray) -> Dict[int, Dict[str, int]]:
        """
        Count normal pieces and kings for each player on a given board.
        Returns: {1: {'pieces': .., 'kings': ..}, -1: {...}}
        """
        return {
            1: {"pieces": int(np.sum(board == 1)), "kings": int(np.sum(board == 2))},
            -1: {"pieces": int(np.sum(board == -1)), "kings": int(np.sum(board == -2))},
        }

    def _count_pieces(self) -> Tuple[int, int]:
        """Total material (men + kings) for each player on the current board."""
        b = self.board.get_state()
        stats = self._piece_and_king_counts_for(b)
        p1 = stats[1]["pieces"] + stats[1]["kings"]
        p2 = stats[-1]["pieces"] + stats[-1]["kings"]
        return p1, p2

    def piece_stats(self) -> Dict[int, Dict[str, int]]:
        """Public helper: piece and king counts for both players on the current board."""
        return self._piece_and_king_counts_for(self.board.get_state())

    def _check_game_over(self) -> Tuple[bool, int]:
        p1, p2 = self._count_pieces()

        if p1 == 0 and p2 == 0:
            return True, 0
        if p1 == 0:
            return True, -1
        if p2 == 0:
            return True, 1

        moves = CheckersRules.get_legal_moves(
            self.board.get_state(),
            self.current_player,
            self.force_capture_from,
        )
        if len(moves) == 0:
            return True, -self.current_player

        return False, 0

    def check_game_over(self) -> Tuple[bool, int]:
        """Public alias for _check_game_over."""
        return self._check_game_over()

    # ------------------------------------------------------------------
    # Move format utilities
    # ------------------------------------------------------------------
    def _is_capture_step(self, action: Any) -> bool:
        """
        Check if `action` is a single capture step of the form (start, landing, jumped),
        where each element is a (row, col) pair.
        """
        return (
            isinstance(action, (list, tuple))
            and len(action) == 3
            and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in action)
        )

    def _normalize_action(self, action: Any) -> Any:
        # Accept a single capture step: (start, landing, jumped)
        if self._is_capture_step(action):
            return tuple(tuple(p) for p in action)

        # Accept legacy wrapper: [ (start, landing, jumped) ]
        if isinstance(action, (list, tuple)) and len(action) == 1 and self._is_capture_step(action[0]):
            return tuple(tuple(p) for p in action[0])

        # If someone passes a multi-step list, keep it as-is (it will be treated illegal)
        if isinstance(action, (list, tuple)) and action and self._is_capture_step(action[0]) and len(action) > 1:
            return action

        # Simple move ((r1, c1), (r2, c2))
        if isinstance(action, (list, tuple)) and len(action) == 2:
            return (tuple(action[0]), tuple(action[1]))

        return action

    # ------------------------------------------------------------------
    # Capture handling
    # ------------------------------------------------------------------
    def _apply_capture_sequence(self, steps: List[CaptureStep], player: int) -> Tuple[Tuple[int, int], int, bool]:
        """
        Apply one or more capture steps.
        WCDF: if a MAN reaches the king-row by capture, it is crowned but the move ends immediately.
        Returns (last_pos, captured_count, continue_flag).
        """
        last_pos: Optional[Tuple[int, int]] = None
        captured_count = 0

        for start, landing, jumped in steps:
            sr, sc = start
            lr, lc = landing
            jr, jc = jumped

            piece_before = self.board.board[sr, sc]  # 1/-1 man, 2/-2 king
            self.board.move_piece(sr, sc, lr, lc)
            self.board.board[jr, jc] = 0

            last_pos = (lr, lc)
            captured_count += 1

            # WCDF: crowning during a capture ends the move (no continuation jumps).
            if abs(piece_before) == 1 and ((player == 1 and lr == 7) or (player == -1 and lr == 0)):
                self.force_capture_from = None
                return last_pos, captured_count, False

        safe_last_pos = last_pos if last_pos is not None else (-1, -1)

        more_captures = CheckersRules.capture_steps(
            self.board.get_state(),
            player,
            forced_from=safe_last_pos,
        )

        if more_captures:
            self.force_capture_from = safe_last_pos
            return safe_last_pos, captured_count, True

        self.force_capture_from = None
        return safe_last_pos, captured_count, False

    # ------------------------------------------------------------------
    # OpenAI-Gym-like API
    # ------------------------------------------------------------------
    def step(self, action: Any):
        """
        Apply an action and return (next_state, reward, done, info) with Gen 7 rewards.
        """
        if self.done:
            # Return the stored terminal winner (not always 0).
            return self.board.get_state(), 0.0, True, {"winner": self.winner, "continue": False, "from": None}

        player = self.current_player
        info: Dict[str, Any] = {"winner": 0, "continue": False, "from": None}

        normalized = self._normalize_action(action)

        legal_moves = self.get_legal_moves()
        if normalized not in legal_moves:
            self.done = True
            self.winner = -player
            info["winner"] = self.winner
            return self.board.get_state(), -1.0, True, info

        captured_count = 0
        promotion_happened = False
        is_man_move = False

        # -----------------------
        # 1) Execute the action
        # -----------------------
        if self._is_capture_step(normalized):
            # Capture sequence (possibly length 1)
            (r_start, c_start) = normalized[0]
            piece_before = self.board.board[r_start, c_start]

            # Detect Man move
            if abs(piece_before) == 1:
                is_man_move = True

            last_pos, count, must_continue = self._apply_capture_sequence([normalized], player)
            captured_count = count

            if last_pos:
                (r_end, c_end) = last_pos
                piece_after = self.board.board[r_end, c_end]
                if abs(piece_before) == 1 and abs(piece_after) == 2:
                    promotion_happened = True

            if must_continue:
                # IMPORTANT: same player moves again; do NOT flip current_player; do NOT increment move_count.
                info["continue"] = True
                info["from"] = last_pos
                return self.board.get_state(), 0.0, False, info

            # Turn ended (capture chain finished)
            self.current_player *= -1

        elif (
            isinstance(normalized, tuple)
            and len(normalized) == 2
            and isinstance(normalized[0], tuple)
            and isinstance(normalized[1], tuple)
            and len(normalized[0]) == 2
            and len(normalized[1]) == 2
        ):
            # Simple move
            (r1, c1), (r2, c2) = normalized
            piece_before = self.board.board[r1, c1]

            # Detect Man move
            if abs(piece_before) == 1:
                is_man_move = True

            self.board.move_piece(r1, c1, r2, c2)

            piece_after = self.board.board[r2, c2]
            if abs(piece_before) == 1 and abs(piece_after) == 2:
                promotion_happened = True

            self.current_player *= -1

        else:
            # Malformed / illegal action: force immediate loss for the acting player
            self.done = True
            self.winner = -player
            info["winner"] = self.winner
            return self.board.get_state(), -1.0, True, info

        # -----------------------
        # 2) Turn ended: update counters and check terminal
        # -----------------------
        self.move_count += 1

        # FIX: Moving a Man is also progress!
        progress = (captured_count > 0) or promotion_happened or is_man_move
        
        if progress:
            self.no_progress = 0
        else:
            self.no_progress += 1

        done, winner = self._check_game_over()

        if not done and self.no_progress >= self.no_progress_limit:
            done = True
            winner = 0

        # Draw by move cap (applies after a completed turn)
        if not done and self.move_count >= self.max_moves:
            done = True
            winner = 0

        self.done = done
        self.winner = winner
        info["winner"] = winner

        # -----------------------
        # 3) Reward shaping
        # -----------------------
        reward = -0.0001  # Living tax default

        if done:
            if winner == player:
                reward = 1.0
            elif winner == -player:
                reward = -1.0
            else:
                reward = 0.0
        else:
            if captured_count >= 2:
                reward = 0.01
            elif captured_count == 1:
                reward = 0.001

        return self.board.get_state(), reward, done, info

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------
    def legal_moves(self):
        if self.done:
            return []
        return CheckersRules.get_legal_moves(self.board.get_state(), self.current_player, self.force_capture_from)

    def get_legal_moves(self):
        return self.legal_moves()

    def render(self):
        self.board.print_board()