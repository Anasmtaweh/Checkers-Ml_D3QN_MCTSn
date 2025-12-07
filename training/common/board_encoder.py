import numpy as np
import torch
from typing import Optional, Dict, Tuple


class CheckersBoardEncoder:
    def encode(self, board: np.ndarray, player: int = 1, info: Optional[Dict] = None) -> torch.Tensor:
        """
        Convert an 8x8 checkers board + metadata into a (12, 8, 8) torch.FloatTensor.
        """
        info = info or {}
        planes = torch.zeros((12, 8, 8), dtype=torch.float32)

        # Piece planes
        planes[0] = torch.from_numpy((board == 1).astype(np.float32))   # Red men
        planes[1] = torch.from_numpy((board == 2).astype(np.float32))   # Red kings
        planes[2] = torch.from_numpy((board == -1).astype(np.float32))  # White men
        planes[3] = torch.from_numpy((board == -2).astype(np.float32))  # White kings

        # Turn planes
        if player == 1:
            planes[4].fill_(1.0)
        elif player == -1:
            planes[5].fill_(1.0)

        # Capture availability
        if info.get("can_capture", False):
            planes[6].fill_(1.0)

        # Repetition count (normalized)
        if "repetition" in info:
            rep_val = min(info.get("repetition", 0), 10) / 10.0
            planes[7].fill_(float(rep_val))

        # Move count (normalized)
        if "move_count" in info:
            move_val = min(info.get("move_count", 0), 100) / 100.0
            planes[8].fill_(float(move_val))

        # Last move origin/destination
        last_move: Optional[Tuple[int, int, int, int]] = info.get("last_move")
        if last_move and len(last_move) == 4:
            r_from, c_from, r_to, c_to = last_move
            if 0 <= r_from < 8 and 0 <= c_from < 8:
                planes[9, r_from, c_from] = 1.0
            if 0 <= r_to < 8 and 0 <= c_to < 8:
                planes[10, r_to, c_to] = 1.0

        # Plane 11 reserved (zeros)
        return planes


if __name__ == "__main__":
    dummy_board = np.zeros((8, 8), dtype=int)
    dummy_board[2, 1] = 1   # Red man
    dummy_board[5, 2] = -1  # White man

    encoder = CheckersBoardEncoder()
    encoded = encoder.encode(dummy_board, player=1, info={"can_capture": False})
    print("Encoded shape:", encoded.shape)
    print("Red men plane sum:", encoded[0].sum().item())
    print("Red turn plane unique values:", torch.unique(encoded[4]))
