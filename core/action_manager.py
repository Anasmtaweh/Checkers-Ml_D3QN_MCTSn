import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

Move = Tuple[Tuple[int, int], Tuple[int, int]]

class ActionManager:
    """
    Maps checkers moves to fixed integer indices for neural network output.
    
    Pre-calculates all theoretically possible moves on an 8x8 board:
    - Simple diagonal moves
    - Jump moves (treating as (start, landing) pairs, ignoring jumped piece)
    
    Provides bidirectional mapping and legal action masking.
    """
    
    def __init__(self, device: Union[str, torch.device] = "cpu"):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.move_to_id: Dict[Move, int] = {}
        self.id_to_move: Dict[int, Move] = {}
        self.action_dim: int = 0
        
        self._build_universal_action_space()
    
    def _build_universal_action_space(self):
        """
        Pre-calculate every theoretically possible move on an 8x8 board.
        Assigns unique integer IDs to each (start, landing) pair.
        Only includes moves whose landing squares are playable dark squares.
        """
        moves = []
        
        # Iterate through all board positions
        for r in range(8):
            for c in range(8):
                # Only consider playable squares (dark squares in checkers)
                if (r + c) % 2 == 1:
                    # Simple diagonal moves (1 square away)
                    for dr in [-1, 1]:
                        for dc in [-1, 1]:
                            nr, nc = r + dr, c + dc
                            # Only include if landing square is valid and on dark square
                            if 0 <= nr < 8 and 0 <= nc < 8 and (nr + nc) % 2 == 1:
                                move = ((r, c), (nr, nc))
                                moves.append(move)
                    
                    # Jump moves (2 squares away diagonally)
                    # These represent (start, landing) pairs for captures
                    for dr in [-2, 2]:
                        for dc in [-2, 2]:
                            nr, nc = r + dr, c + dc
                            # Only include if landing square is valid and on dark square
                            if 0 <= nr < 8 and 0 <= nc < 8 and (nr + nc) % 2 == 1:
                                move = ((r, c), (nr, nc))
                                moves.append(move)
        
        # Remove duplicates and sort for consistency
        moves = list(set(moves))
        moves.sort()
        
        # Build bidirectional mappings
        for idx, move in enumerate(moves):
            self.move_to_id[move] = idx
            self.id_to_move[idx] = move
        
        self.action_dim = len(moves)
        
        print(f"ActionManager initialized: {self.action_dim} total actions")
    
    def get_action_id(self, move: Move) -> int:
        """
        Convert a move to its action ID.
        
        Args:
            move: ((r1, c1), (r2, c2)) tuple
            
        Returns:
            Integer action ID, or -1 if move not in action space
        """
        # Normalize move format to Match the dict key type
        normalized: Move = (tuple(move[0]), tuple(move[1]))  # type: ignore
        return int(self.move_to_id.get(normalized, -1))
    
    def get_move_from_id(self, action_id: int) -> Move:
        """
        Convert an action ID back to a move.
        
        Args:
            action_id: Integer action index
            
        Returns:
            Move tuple ((r1, c1), (r2, c2))
        """
        return self.id_to_move.get(action_id, ((-1, -1), (-1, -1)))
    
    def make_legal_action_mask(self, legal_moves: List[Any], player: int = 1) -> torch.Tensor:
        """
        Create a boolean mask tensor indicating which actions are legal.
        
        Args:
            legal_moves: List of legal moves from the environment.
                Can be simple moves or single capture steps.
            player: Current player (1 or -1). If -1, moves are flipped to canonical perspective.
                
        Returns:
            Boolean tensor of shape [action_dim] where True indicates legal actions.
        """
        mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
        
        for move in legal_moves:
            # Extract (start, landing) pair from various move formats
            start_landing = self._extract_start_landing(move)
            
            if start_landing is not None:
                # Canonicalize: If player is -1 (Black), flip move to match P1 perspective
                if player == -1:
                    start_landing = self.flip_move(start_landing)
                
                action_id = self.get_action_id(start_landing)
                if action_id >= 0:
                    mask[action_id] = True
        
        return mask
    
    def _extract_start_landing(self, move: Any) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Extract (start, landing) pair from move formats.
        
        Handles:
        - Simple moves: ((r1, c1), (r2, c2))
        - Single capture step: ((r1, c1), (r2, c2), (jr, jc))
        
        Returns:
            ((start_r, start_c), (landing_r, landing_c)) or None
        """
        if not move:
            return None
        
        # Case 1: Single capture step (start, landing, jumped)
        if isinstance(move, (tuple, list)) and len(move) == 3:
            # Check if third element looks like coordinates
            if isinstance(move[2], (tuple, list)) and len(move[2]) == 2:
                return (tuple(move[0]), tuple(move[1]))
        
        # Case 2: Simple move ((r1, c1), (r2, c2))
        if isinstance(move, (tuple, list)) and len(move) == 2:
            if isinstance(move[0], (tuple, list)) and isinstance(move[1], (tuple, list)):
                if len(move[0]) == 2 and len(move[1]) == 2:
                    return (tuple(move[0]), tuple(move[1]))
        
        return None
    
    def to(self, device):
        """Move to a different device."""
        self.device = torch.device(device)
        return self

    def flip_move(self, move: Move) -> Move:
        """
        Flip a move 180 degrees (for P2 perspective correction).
        (r, c) -> (7-r, 7-c)
        
        Args:
            move: ((r1, c1), (r2, c2))
            
        Returns:
            Flipped move tuple
        """
        start, end = move
        return (
            (7 - start[0], 7 - start[1]),
            (7 - end[0], 7 - end[1])
        )
