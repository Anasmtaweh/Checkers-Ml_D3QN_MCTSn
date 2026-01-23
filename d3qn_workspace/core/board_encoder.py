import torch
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

class CheckersBoardEncoder:
    """
    Converts raw 8x8 numpy board into CNN-friendly PyTorch tensor.
    
    Updated: Now uses 6 channels to provide context for mandatory multi-jumps.
    """
    
    def __init__(self):
        pass
    
    def encode(
        self, 
        board: np.ndarray, 
        player: int = 1, 
        force_move_from: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Encode board state into 6-channel tensor.
        
        Channel 0-3: Pieces (My Men, My Kings, Enemy Men, Enemy Kings)
        Channel 4: Tempo/Identity (Constant plane)
        Channel 5: Forced Move Mask (1.0 at the square of the forced piece)
        """
        # Canonicalize: Make the network always see itself as Player 1
        canonical_board = self.canonicalize_board(board, player)
        
        # Create 6 feature planes (Changed from 5 to 6)
        planes = np.zeros((6, 8, 8), dtype=np.float32)
        
        # Channel 0: My men (value = 1 after canonicalization)
        planes[0] = (canonical_board == 1).astype(np.float32)
        
        # Channel 1: My kings (value = 2 after canonicalization)
        planes[1] = (canonical_board == 2).astype(np.float32)
        
        # Channel 2: Enemy men (value = -1 after canonicalization)
        planes[2] = (canonical_board == -1).astype(np.float32)
        
        # Channel 3: Enemy kings (value = -2 after canonicalization)
        planes[3] = (canonical_board == -2).astype(np.float32)
        
        # Channel 4: Tempo/identity plane
        tempo_value = 0.0 if player == 1 else 1.0
        planes[4] = np.full((8, 8), tempo_value, dtype=np.float32)

        # Channel 5: Forced Move Mask (The context fix)
        if force_move_from is not None:
            r, c = force_move_from
            # If we are P2, the board was rotated 180, so flip the coordinate!
            if player == -1:
                r, c = 7 - r, 7 - c
            if 0 <= r < 8 and 0 <= c < 8:
                planes[5, r, c] = 1.0
        
        return torch.from_numpy(planes)
    
    def canonicalize_board(self, board: np.ndarray, player: int) -> np.ndarray:
        if player == 1:
            return board.copy()
        elif player == -1:
            # Rotate 180 degrees and swap piece identifiers
            rotated = np.rot90(board, k=2)
            return -rotated
        else:
            raise ValueError(f"Invalid player value: {player}")
    
    def decode(self, encoded: torch.Tensor, player: int = 1) -> np.ndarray:
        if encoded.dim() == 4:
            encoded = encoded[0]
        
        planes = encoded.cpu().numpy() if isinstance(encoded, torch.Tensor) else encoded
        
        canonical_board = np.zeros((8, 8), dtype=int)
        canonical_board[planes[0] > 0.5] = 1
        canonical_board[planes[1] > 0.5] = 2
        canonical_board[planes[2] > 0.5] = -1
        canonical_board[planes[3] > 0.5] = -2
        
        if player == 1:
            return canonical_board
        else:
            flipped = -canonical_board
            return np.rot90(flipped, k=2)
    
    def batch_encode(
        self, 
        boards: List[np.ndarray], 
        players: List[int],
        force_moves: Optional[List[Optional[Tuple[int, int]]]] = None
    ) -> torch.Tensor:
        # Resolve force_moves to a concrete list to satisfy type checker
        use_force_moves: List[Optional[Tuple[int, int]]]
        if force_moves is None:
            use_force_moves = [None] * len(boards)
        else:
            use_force_moves = force_moves
        
        encoded_list = [
            self.encode(board, player, fm)
            for board, player, fm in zip(boards, players, use_force_moves)
        ]
        
        return torch.stack(encoded_list, dim=0)