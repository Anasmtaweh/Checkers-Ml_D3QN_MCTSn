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

    def __init__(self):
        self.board = CheckersBoard()
        self.current_player = 1  # player to move
        self.done = False
        # When a capture chain is in progress, this is the square that must continue
        self.force_capture_from: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def reset(self):
        """Reset to starting position and return the board array."""
        self.board.reset()
        self.current_player = 1
        self.done = False
        self.force_capture_from = None
        return self.board.get_state()

    def _piece_and_king_counts_for(self, board: np.ndarray) -> Dict[int, Dict[str, int]]:
        """
        Count normal pieces and kings for each player on a given board.
        Returns: {1: {'pieces': .., 'kings': ..}, -1: {...}}
        """
        return {
            1: {
                "pieces": int(np.sum(board == 1)),
                "kings": int(np.sum(board == 2)),
            },
            -1: {
                "pieces": int(np.sum(board == -1)),
                "kings": int(np.sum(board == -2)),
            },
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
        """
        Returns (done, winner) where winner is:
          1  -> player 1 wins
         -1  -> player -1 wins
          0  -> draw / not decided
        """
        p1, p2 = self._count_pieces()
        if p1 == 0 and p2 == 0:
            return True, 0
        if p1 == 0:
            return True, -1
        if p2 == 0:
            return True, 1

        # If side to move has no legal moves, they lose
        moves = CheckersRules.get_legal_moves(self.board.get_state(), self.current_player, self.force_capture_from)
        if len(moves) == 0:
            winner = -self.current_player
            return True, winner

        return False, 0

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
        """
        Convert an incoming action (from agent or environment) into one of:
          - list[CaptureStep]   -> capture chain
          - Move               -> simple move ((r1, c1), (r2, c2))
        Capture action can be a single step or a full sequence (list of steps).
        """
        # A single capture step (start, landing, jumped)
        if self._is_capture_step(action):
            return [tuple(tuple(p) for p in action)]  # [ (start, landing, jumped) ]

        # A full capture sequence: [ (start, landing, jumped), ... ]
        if isinstance(action, (list, tuple)) and action and self._is_capture_step(action[0]):
            return [tuple(tuple(p) for p in step) for step in action]

        # Simple move ((r1, c1), (r2, c2))
        if isinstance(action, (list, tuple)) and len(action) == 2:
            return (tuple(action[0]), tuple(action[1]))

        # Unknown / malformed -> just return as-is, will likely be rejected by rules
        return action

    # ------------------------------------------------------------------
    # Capture handling
    # ------------------------------------------------------------------
    def _apply_capture_sequence(self, steps: List[CaptureStep], player: int) -> Tuple[Tuple[int, int], int, bool]:
        """
        Apply one or more capture steps.
        Returns (last_pos, captured_count, continue_flag).
        """
        last_pos: Optional[Tuple[int, int]] = None

        # Apply each capture step in sequence
        for start, landing, jumped in steps:
            sr, sc = start
            lr, lc = landing
            jr, jc = jumped

            # Move the capturing piece
            self.board.move_piece(sr, sc, lr, lc)
            # Remove the captured piece
            self.board.board[jr, jc] = 0

            last_pos = (lr, lc)

        captured_count = len(steps)

        # Check if another capture is possible from last_pos (capture chains)
        more_captures = CheckersRules.capture_sequences(
            self.board.get_state(),
            player,
            start_pos=last_pos,
        )

        # Guarantee last_pos is always a Tuple[int, int] for type-checkers
        safe_last_pos = last_pos if last_pos is not None else (-1, -1)

        if more_captures:
            # Must continue chain with the same piece; do not change player yet.
            self.force_capture_from = safe_last_pos
            return safe_last_pos, captured_count, True

        # No more captures; chain ends and turn will switch.
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
            return self.board.get_state(), 0.0, True, {"winner": 0}

        player = self.current_player
        info: Dict[str, Any] = {"winner": 0, "continue": False, "from": None}

        # 1. Parse and Execute Move
        normalized = self._normalize_action(action)
        
        # Track if we captured something
        captured_count = 0

        if isinstance(normalized, list):
            # It's a capture sequence
            last_pos, count, must_continue = self._apply_capture_sequence(normalized, player)
            captured_count = count

            if must_continue:
                self.done = False
                info["continue"] = True
                info["from"] = last_pos
                # Intermediate reward for multi-jump progress?
                # Usually better to wait until turn ends, but if you want:
                return self.board.get_state(), 0.0, False, info

            self.current_player *= -1
        else:
            # Simple move
            (r1, c1), (r2, c2) = normalized
            self.board.move_piece(r1, c1, r2, c2)
            self.current_player *= -1

        # 2. Check Game Over
        done, winner = self._check_game_over()
        self.done = done
        info["winner"] = winner

        # 3. Calculate "Gen 7" Rewards directly here
        reward = -0.0001  # Living Tax (Default)

        if done:
            if winner == player:
                reward = 1.0   # WIN
            elif winner == -player:
                reward = -1.0  # LOSS
            else:
                reward = 0.0   # DRAW
        else:
            # If not done, check for captures
            if captured_count >= 2:
                reward = 0.01  # Multi-jump
            elif captured_count == 1:
                reward = 0.001 # Single jump

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
