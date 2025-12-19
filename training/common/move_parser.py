"""
Move parser utility for converting environment legal moves to normalized format.
Works in conjunction with ActionManager to handle various move representations.
"""

from typing import List, Tuple, Dict, Any

Move = Tuple[Tuple[int, int], Tuple[int, int]]

def parse_legal_moves(legal_moves: List[Any], action_manager) -> Tuple[List[Move], Dict[int, Any]]:
    """
    Parse legal moves from environment and create mapping to action indices.
    
    Args:
        legal_moves: List of legal moves from environment (various formats)
        action_manager: ActionManager instance for move-to-ID conversion
        
    Returns:
        Tuple of:
        - normalized_moves: List of (start, landing) tuples
        - mapping: Dict[action_id -> original_env_move]
    """
    normalized_moves = []
    mapping = {}
    
    for env_move in legal_moves:
        # Extract (start, landing) pair
        start_landing = action_manager._extract_start_landing(env_move)
        
        if start_landing is not None:
            # Get action ID for this move
            action_id = action_manager.get_action_id(start_landing)
            
            if action_id >= 0:
                normalized_moves.append(start_landing)
                mapping[action_id] = env_move
    
    return normalized_moves, mapping


def is_capture_move(move: Any) -> bool:
    """
    Determine if a move is a capture.
    
    Args:
        move: Move in any format
        
    Returns:
        True if move is a capture, False otherwise
    """
    # Capture sequence: list of steps
    if isinstance(move, list) and move:
        return True
    
    # Single capture step: (start, landing, jumped)
    if isinstance(move, (tuple, list)) and len(move) == 3:
        if isinstance(move[2], (tuple, list)) and len(move[2]) == 2:
            return True
    
    return False


def get_move_distance(move: Move) -> int:
    """
    Calculate Manhattan distance of a move.
    
    Args:
        move: ((r1, c1), (r2, c2)) tuple
        
    Returns:
        Manhattan distance between start and end
    """
    (r1, c1), (r2, c2) = move
    return abs(r2 - r1) + abs(c2 - c1)


def normalize_move_format(move: Any) -> Any:
    """
    Normalize move to consistent tuple format.
    
    Args:
        move: Move in any format (list or tuple)
        
    Returns:
        Move with all nested lists converted to tuples
    """
    if isinstance(move, list):
        if not move:
            return move
        # Capture sequence
        if isinstance(move[0], (list, tuple)) and len(move[0]) == 3:
            return [tuple(tuple(p) if isinstance(p, list) else p for p in step) 
                    for step in move]
    
    if isinstance(move, (tuple, list)):
        if len(move) == 2:
            # Simple move
            return (tuple(move[0]) if isinstance(move[0], list) else move[0],
                    tuple(move[1]) if isinstance(move[1], list) else move[1])
        elif len(move) == 3:
            # Single capture
            return (tuple(move[0]) if isinstance(move[0], list) else move[0],
                    tuple(move[1]) if isinstance(move[1], list) else move[1],
                    tuple(move[2]) if isinstance(move[2], list) else move[2])
    
    return move
