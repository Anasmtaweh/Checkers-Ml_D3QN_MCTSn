import torch
import numpy as np
from typing import Optional, Dict, Any, List

class CheckersBoardEncoder:
    """
    Converts raw 8x8 numpy board into CNN-friendly PyTorch tensor.
    
    Key Features:
    - Canonicalization: Always presents board from Player 1 perspective
    - If current player is -1, rotates board 180° and swaps piece IDs
    - Outputs 5 feature planes: my men, my kings, enemy men, enemy kings, tempo
    
    Output Shape: (5, 8, 8) or (Batch, 5, 8, 8)
    """
    
    def __init__(self):
        pass
    
    def encode(
        self, 
        board: np.ndarray, 
        player: int = 1, 
        info: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Encode board state into 5-channel tensor.
        
        Args:
            board: Raw 8x8 numpy array with values in {-2, -1, 0, 1, 2}
                   1 = Player 1 man, 2 = Player 1 king
                  -1 = Player 2 man, -2 = Player 2 king
            player: Current player (1 or -1)
            info: Optional info dict (unused, for API compatibility)
            
        Returns:
            Tensor of shape (5, 8, 8) with:
                Channel 0: My men (1.0 where present)
                Channel 1: My kings (1.0 where present)
                Channel 2: Enemy men (1.0 where present)
                Channel 3: Enemy kings (1.0 where present)
                Channel 4: Tempo/identity plane (0.0 for P1, 1.0 for P2)
        """
        # Canonicalize: Make the network always see itself as Player 1
        canonical_board = self._canonicalize_board(board, player)
        
        # Create 5 feature planes
        planes = np.zeros((5, 8, 8), dtype=np.float32)
        
        # Channel 0: My men (value = 1 after canonicalization)
        planes[0] = (canonical_board == 1).astype(np.float32)
        
        # Channel 1: My kings (value = 2 after canonicalization)
        planes[1] = (canonical_board == 2).astype(np.float32)
        
        # Channel 2: Enemy men (value = -1 after canonicalization)
        planes[2] = (canonical_board == -1).astype(np.float32)
        
        # Channel 3: Enemy kings (value = -2 after canonicalization)
        planes[3] = (canonical_board == -2).astype(np.float32)
        
        # Channel 4: Tempo/identity plane
        # 0.0 if we are physically Player 1 (starter)
        # 1.0 if we are physically Player 2 (follower/reactive side)
        tempo_value = 0.0 if player == 1 else 1.0
        planes[4] = np.full((8, 8), tempo_value, dtype=np.float32)
        
        return torch.from_numpy(planes)
    
    def _canonicalize_board(self, board: np.ndarray, player: int) -> np.ndarray:
        """
        Transform board so the current player is always represented as Player 1.
        
        If current player is -1 (Player 2):
            1. Rotate board 180 degrees
            2. Swap piece identifiers: -1 becomes 1, -2 becomes 2, etc.
        
        Args:
            board: Original 8x8 board
            player: Current player (1 or -1)
            
        Returns:
            Canonicalized 8x8 board where current player is represented as 1/2
        """
        if player == 1:
            # Already viewing as Player 1, no transformation needed
            return board.copy()
        else:
            # Player is -1, need to flip perspective
            # Step 1: Rotate 180 degrees (flip both vertically and horizontally)
            rotated = np.rot90(board, k=2)
            
            # Step 2: Swap piece identifiers (multiply by -1)
            # This makes -1 -> 1, -2 -> 2, 1 -> -1, 2 -> -2
            canonicalized = -rotated
            
            return canonicalized
    
    def decode(
        self, 
        encoded: torch.Tensor, 
        player: int = 1
    ) -> np.ndarray:
        """
        Convert encoded tensor back to raw board format (for debugging/visualization).
        
        Args:
            encoded: Tensor of shape (5, 8, 8) or (Batch, 5, 8, 8)
            player: The player this encoding was for (1 or -1)
            
        Returns:
            Reconstructed 8x8 numpy array
        """
        # Handle batch dimension
        if encoded.dim() == 4:
            encoded = encoded[0]
        
        # Convert to numpy
        if isinstance(encoded, torch.Tensor):
            planes = encoded.cpu().numpy()
        else:
            planes = encoded
        
        # Reconstruct canonical board
        canonical_board = np.zeros((8, 8), dtype=int)
        canonical_board[planes[0] > 0.5] = 1   # My men
        canonical_board[planes[1] > 0.5] = 2   # My kings
        canonical_board[planes[2] > 0.5] = -1  # Enemy men
        canonical_board[planes[3] > 0.5] = -2  # Enemy kings
        
        # De-canonicalize if needed
        if player == 1:
            return canonical_board
        else:
            # Reverse the transformation: multiply by -1, then rotate 180
            flipped = -canonical_board
            return np.rot90(flipped, k=2)
    
    def batch_encode(
        self, 
        boards: List[np.ndarray], 
        players: List[int],
        infos: Optional[List[Dict[str, Any]]] = None
    ) -> torch.Tensor:
        """
        Encode multiple boards into a batched tensor.
        
        Args:
            boards: List of 8x8 numpy arrays
            players: List of current player for each board
            infos: Optional list of info dicts
            
        Returns:
            Tensor of shape (Batch, 5, 8, 8)
        """
        if infos is None:
            infos = [None] * len(boards)  # type: ignore
        
        assert infos is not None
        encoded_list = [
            self.encode(board, player, info)
            for board, player, info in zip(boards, players, infos)
        ]
        
        return torch.stack(encoded_list, dim=0)
