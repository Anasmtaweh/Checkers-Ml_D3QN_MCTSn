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

    # --- Material delta shaping (Option A: end-of-turn only) ---
    MATERIAL_MEN_W = 1.0
    MATERIAL_KING_W = 1.5
    MATERIAL_SCALE = 1.0

    def _material_score(self, board: np.ndarray, player: int) -> float:
        stats = self._piece_and_king_counts_for(board)
        men = stats[player]["pieces"]
        kings = stats[player]["kings"]
        return men * self.MATERIAL_MEN_W + kings * self.MATERIAL_KING_W

    def _material_delta_reward(self, before: np.ndarray, after: np.ndarray, player: int) -> float:
        me_before = self._material_score(before, player)
        me_after = self._material_score(after, player)
        opp = -player
        opp_before = self._material_score(before, opp)
        opp_after = self._material_score(after, opp)

        delta_me = me_after - me_before
        delta_opp = opp_after - opp_before
        return self.MATERIAL_SCALE * (delta_me - delta_opp)

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
    def _apply_capture_sequence(self, steps: List[CaptureStep], player: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Apply one or more capture steps.
        Returns (last_pos, total_reward, continue_flag).

        Reward components here:
          +10  per captured piece
          -1   per jump (baseline move cost)
          +4   per extra jump in the same turn (multi-jump bonus)
        """
        total_reward = 0.0
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
            total_reward += 10.0   # captured one piece
            total_reward -= 1.0    # small "time" penalty

        # Extra shaping for multi-jump:
        num_captures = len(steps)
        if num_captures > 1:
            # First capture gets the base +10, each additional capture gets a bit extra.
            total_reward += (num_captures - 1) * 4.0

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
            return safe_last_pos, total_reward, True

        # No more captures; chain ends and turn will switch.
        self.force_capture_from = None
        return safe_last_pos, total_reward, False

    # ------------------------------------------------------------------
    # Positional reward shaping
    # ------------------------------------------------------------------
    def _player_squares(self, board: np.ndarray, player: int) -> List[Tuple[int, int]]:
        """Return all (row, col) squares occupied by `player` (men or kings)."""
        positions: List[Tuple[int, int]] = []
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                if np.sign(board[r, c]) == player and board[r, c] != 0:
                    positions.append((r, c))
        return positions

    def _positional_shaping(
        self,
        before: np.ndarray,
        after: np.ndarray,
        player: int,
    ) -> float:
        """
        Small, smooth rewards on top of the main ones to encourage:
          - king promotion
          - occupying central squares
          - avoiding immediately hanging the moved piece
        """
        reward = 0.0
        stats_before = self._piece_and_king_counts_for(before)
        stats_after = self._piece_and_king_counts_for(after)

        me = player
        opp = -player

        # 1) King promotion bonus (per new king)
        new_kings = stats_after[me]["kings"] - stats_before[me]["kings"]
        if new_kings > 0:
            reward += 5.0 * new_kings

        # 2) Identify the square where our moved piece ended up
        before_sq = set(self._player_squares(before, me))
        after_sq = set(self._player_squares(after, me))
        added = list(after_sq - before_sq)

        moved_to: Optional[Tuple[int, int]] = None
        if len(added) == 1:
            moved_to = added[0]

            r, c = moved_to

            # 2a) Central control: encourage staying near the center
            # Board is 8x8, indices 0..7; treat rows 2..5 and cols 2..5 as "center"
            if 2 <= r <= 5 and 2 <= c <= 5:
                reward += 0.5
            # Slight penalty for sitting on the extreme back ranks (excluding promotion)
            if r in (0, 7):
                reward -= 0.2

        # 3) Avoid immediately hanging the moved piece:
        #    Look one ply ahead from opponent's perspective; if any capture
        #    explicitly jumps over our moved square, apply a small penalty.
        if moved_to is not None:
            enemy_moves = CheckersRules.get_legal_moves(after, opp, None)
            in_danger = False
            for mv in enemy_moves:
                norm = self._normalize_action(mv)
                if isinstance(norm, list):
                    # capture chain -> list of (start, landing, jumped)
                    for (_, _, jumped) in norm:
                        if tuple(jumped) == moved_to:
                            in_danger = True
                            break
                if in_danger:
                    break

            if in_danger:
                # OLD: reward -= 1.0
                # NEW: Reduced penalty.
                # We want the agent to learn safety from Q-loss, not be terrified to move.
                reward -= 0.1

        return reward

    # ------------------------------------------------------------------
    # OpenAI-Gym-like API
    # ------------------------------------------------------------------
    def step(self, action: Any):
        """
        Apply an action and return (next_state, reward, done, info).

        Reward components:
          -1    per move (time penalty)
          +10   per captured piece (inside _apply_capture_sequence)
          +4    per extra jump in a multi-jump (inside _apply_capture_sequence)
          +5    per new king
          +0.5  for moving into the central area
          -0.2  for ending turn on back rank (non-promotion)
          -1    if the moved piece is immediately capturable next ply
          +50   win, -50 loss at game end
        """
        if self.done:
            return self.board.get_state(), 0.0, True, {"winner": 0}

        # Snapshot before applying the action for shaping
        before_board = np.copy(self.board.get_state())
        player = self.current_player
        info: Dict[str, Any] = {"winner": 0, "continue": False, "from": None}

        normalized = self._normalize_action(action)

        # --- Capture path (possibly multi-jump) ---
        if isinstance(normalized, list):
            last_pos, reward, must_continue = self._apply_capture_sequence(normalized, player)

            if must_continue:
                # Capture chain is not finished; same player must continue from last_pos.
                self.done = False
                info["continue"] = True
                info["from"] = last_pos
                # We don't apply positional shaping yet because the move isn't finished.
                return self.board.get_state(), reward, False, info

            # Capture chain finished -> switch turn after positional shaping.
            self.current_player *= -1

        # --- Simple move path ---
        else:
            # Expected format: ((r1, c1), (r2, c2))
            (r1, c1), (r2, c2) = normalized
            self.board.move_piece(r1, c1, r2, c2)
            reward = -1.0  # base move penalty
            self.force_capture_from = None
            self.current_player *= -1

        # Apply positional shaping based on before/after board states
        after_board = self.board.get_state()
        reward += self._positional_shaping(before_board, after_board, player)
        # NEW: material delta shaping (end-of-turn)
        reward += self._material_delta_reward(before_board, after_board, player)

        # Check for terminal outcome
        done, winner = self._check_game_over()
        self.done = done
        info["winner"] = winner

        if done:
            if winner == player:
                reward += 50.0
            elif winner == -player:
                reward -= 50.0

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
