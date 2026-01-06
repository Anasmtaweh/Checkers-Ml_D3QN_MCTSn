import numpy as np


class CheckersRules:
    @staticmethod
    def _piece_directions(board, r, c):
        """Return list of row directions this piece can move in."""
        val = board[r, c]
        if abs(val) == 2:  # king can go both ways
            return [1, -1]
        return [1] if val > 0 else [-1]

    @staticmethod
    def _is_opponent(val, player):
        return val != 0 and np.sign(val) == -player

    @staticmethod
    def simple_moves(board, player):
        """Non-capturing moves for the given player."""
        moves = []
        for r in range(8):
            for c in range(8):
                if board[r, c] == player or board[r, c] == 2 * player:
                    for dr in CheckersRules._piece_directions(board, r, c):
                        for dc in (-1, 1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < 8 and 0 <= nc < 8 and board[nr, nc] == 0:
                                moves.append(((r, c), (nr, nc)))
        return moves

    @staticmethod
    def _capture_steps_from(board, r, c, player):
        """
        Return all immediate capture steps from position (r, c).
        Returns list of single steps: ((r_start, c_start), (r_land, c_land), (r_jumped, c_jumped))
        """
        steps = []

        if not (0 <= r < 8 and 0 <= c < 8):
            return steps
        if not (board[r, c] == player or board[r, c] == 2 * player):
            return steps

        for dr in CheckersRules._piece_directions(board, r, c):
            for dc in (-1, 1):
                mid_r, mid_c = r + dr, c + dc
                land_r, land_c = r + 2 * dr, c + 2 * dc

                if not (0 <= mid_r < 8 and 0 <= mid_c < 8 and 0 <= land_r < 8 and 0 <= land_c < 8):
                    continue

                if CheckersRules._is_opponent(board[mid_r, mid_c], player) and board[land_r, land_c] == 0:
                    steps.append(((r, c), (land_r, land_c), (mid_r, mid_c)))

        return steps


    @staticmethod
    def _capture_sequences_from(board, r, c, player):
        """
        Depth-first search for all capture chains starting from (r, c).
        Returns list of sequences, where each sequence is a list of
        (start, landing, jumped) triples.
        """
        sequences = []
        for dr in CheckersRules._piece_directions(board, r, c):
            for dc in (-1, 1):
                mid_r, mid_c = r + dr, c + dc
                land_r, land_c = r + 2 * dr, c + 2 * dc

                if not (0 <= mid_r < 8 and 0 <= mid_c < 8 and 0 <= land_r < 8 and 0 <= land_c < 8):
                    continue

                if CheckersRules._is_opponent(board[mid_r, mid_c], player) and board[land_r, land_c] == 0:
                    # simulate jump
                    new_board = board.copy()
                    new_board[land_r, land_c] = new_board[r, c]
                    new_board[r, c] = 0
                    new_board[mid_r, mid_c] = 0

                    subsequent = CheckersRules._capture_sequences_from(new_board, land_r, land_c, player)
                    step = ((r, c), (land_r, land_c), (mid_r, mid_c))

                    if subsequent:
                        for seq in subsequent:
                            sequences.append([step] + seq)
                    else:
                        sequences.append([step])
        return sequences

    @staticmethod
    def capture_steps(board, player, forced_from=None):
        """
        Return all immediate capture steps for player.
        Each step is a single tuple: ((r_start, c_start), (r_land, c_land), (r_jumped, c_jumped))
        If forced_from is provided, restrict captures to that piece only.
        """
        steps = []
        positions = []
        if forced_from is not None:
            positions = [forced_from]
        else:
            for r in range(8):
                for c in range(8):
                    if board[r, c] == player or board[r, c] == 2 * player:
                        positions.append((r, c))

        for r, c in positions:
            steps.extend(CheckersRules._capture_steps_from(board, r, c, player))
        return steps

    @staticmethod
    def capture_sequences(board, player, start_pos=None):
        """
        Return all capture sequences for player.
        Each sequence is a list of (start, landing, jumped) triples.
        If start_pos is provided, restrict search to that piece (for chain captures).
        """
        captures = []
        positions = []
        if start_pos is not None:
            positions = [start_pos]
        else:
            for r in range(8):
                for c in range(8):
                    if board[r, c] == player or board[r, c] == 2 * player:
                        positions.append((r, c))

        for r, c in positions:
            captures.extend(CheckersRules._capture_sequences_from(board, r, c, player))
        return captures

    @staticmethod
    def get_legal_moves(board, player, forced_from=None):
        """
        Return mandatory captures if any (as single steps), otherwise simple moves.
        forced_from restricts capture search to a single piece for chain moves.
        Returns list of single capture steps: ((r_start, c_start), (r_land, c_land), (r_jumped, c_jumped))
        or simple moves: ((r_start, c_start), (r_land, c_land))
        """
        captures = CheckersRules.capture_steps(board, player, forced_from)
        if captures:
            return captures
        return CheckersRules.simple_moves(board, player)
