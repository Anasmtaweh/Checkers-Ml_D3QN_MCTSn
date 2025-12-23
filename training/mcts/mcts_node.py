import math
import random
import torch
from typing import Optional, List, Tuple

class MCTSNode:
    """
    A node in the MCTS tree.
    """
    def __init__(self, state, parent=None, action=None, player=None):
        self.state = state  # Game state (your CheckersGame object)
        self.parent = parent
        self.action = action  # Action that led to this node
        self.player = player  # Player who made the move to reach this state
        
        self.children: List[MCTSNode] = []
        self.untried_actions: Optional[List[Tuple]] = None  # Will be populated on first visit
        
        self.visits = 0
        self.value = 0.0  # Total reward
        
    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been tried."""
        return len(self.untried_actions) == 0 if self.untried_actions is not None else False
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal game state."""
        return self.state.is_game_over()
    
    def best_child(self, exploration_weight=1.414) -> 'MCTSNode':
        """
        Select best child using UCB1 formula.
        exploration_weight: higher = more exploration (default sqrt(2))
        """
        def ucb1(child):
            if child.visits == 0:
                return float('inf')
            
            exploitation = child.value / child.visits
            exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            return exploitation + exploration
        
        return max(self.children, key=ucb1)
    
    def add_child(self, state, action, player) -> 'MCTSNode':
        """Add a child node for the given action."""
        child = MCTSNode(state, parent=self, action=action, player=player)
        self.children.append(child)
        return child


def select(node: MCTSNode) -> MCTSNode:
    """
    Select a node to expand using UCB1.
    Traverse down the tree until we find a node that's not fully expanded.
    """
    while not node.is_terminal():
        if node.untried_actions is None:
            # First visit - initialize untried actions
            node.untried_actions = node.state.get_valid_moves()
            return node
        
        if not node.is_fully_expanded():
            return node
        
        # Fully expanded - select best child
        node = node.best_child()
    
    return node


def expand(node: MCTSNode) -> MCTSNode:
    """
    Expand the node by trying an untried action.
    Prioritizes capture moves (better moves first).
    Returns the new child node.
    """
    if node.untried_actions is None:
        actions = node.state.get_valid_moves()
        # Sort: capture moves first (longer paths are captures in checkers)
        node.untried_actions = sorted(actions, key=lambda a: len(a) if isinstance(a, list) else 0, reverse=True)
    
    # Local reference for type safety
    untried = node.untried_actions
    
    if not untried:
        return node  # No actions to try or None
    
    # Pick first untried action (prioritized: captures first)
    action = untried.pop(0)
    
    # Create new state
    new_state = node.state.clone()  # Deep copy
    new_state.make_move(action)
    
    # Add child node
    child = node.add_child(new_state, action, new_state.current_player)
    
    return child


def simulate(node: MCTSNode, max_depth=50, eval_model=None, action_manager=None, device=None) -> float:
    """
    Fast random rollout with improved depth and optional neural network evaluation.
    """
    state = node.state.clone()
    depth = 0
    
    while not state.is_game_over() and depth < max_depth:
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            break
        
        action = random.choice(valid_moves)
        state.make_move(action)
        depth += 1
    
    # Use heuristic if didn't reach terminal
    if state.is_game_over():
        winner = state.get_winner()
        if winner == node.player:
            return 1.0
        elif winner == 0:
            return 0.5
        else:
            return 0.0
    else:
        # Try to use neural network evaluation if available
        if eval_model is not None and action_manager is not None and device is not None:
            try:
                return evaluate_with_neural_net(state, node.player, eval_model, action_manager, device)
            except:
                # Fallback to heuristic if neural eval fails
                return evaluate_position(state, node.player)
        else:
            return evaluate_position(state, node.player)


def evaluate_position(state, player) -> float:
    """
    Advanced position evaluation - kings, piece count, board control, and endgame bonuses.
    Heavily weighted toward piece advantage.
    """
    red_pieces = 0
    red_kings = 0
    red_back_row = 0  # Pieces on promotion row
    black_pieces = 0
    black_kings = 0
    black_back_row = 0
    
    board = state.board
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece == 1:
                red_pieces += 1
                if row == 7:  # Back row
                    red_back_row += 1
            elif piece == 2:
                red_pieces += 1
                red_kings += 1
            elif piece == -1:
                black_pieces += 1
                if row == 0:  # Back row
                    black_back_row += 1
            elif piece == -2:
                black_pieces += 1
                black_kings += 1
    
    # CRITICAL: Piece count is the most important factor
    # Kings worth MUCH more than pawns - capture them at all costs
    red_score = red_pieces * 100 + red_kings * 500
    black_score = black_pieces * 100 + black_kings * 500
    
    # Add back row danger bonus (closer to promotion)
    red_score += red_back_row * 80
    black_score += black_back_row * 80
    
    # Add mobility bonus (critical for winning positions)
    # More moves = more options = better position
    if state.current_player == 1:
        red_mobility = len(state.get_valid_moves())
        black_mobility = 0
    else:
        red_mobility = 0
        black_mobility = len(state.get_valid_moves())
    
    red_score += red_mobility * 20
    black_score += black_mobility * 20
    
    # Aggressive endgame bonus: when few pieces remain, advantage is CRITICAL
    total_pieces = red_pieces + black_pieces
    if total_pieces <= 2:  # Critical endgame
        endgame_factor = 5.0
    elif total_pieces <= 4:  # Endgame
        endgame_factor = 3.0
    elif total_pieces <= 8:
        endgame_factor = 1.5
    else:
        endgame_factor = 1.0
    
    red_score *= endgame_factor
    black_score *= endgame_factor
    
    # Normalize to [0, 1]
    total_score = red_score + black_score
    if total_score == 0:
        return 0.5
    
    if player == 1:  # Red
        return min(1.0, red_score / total_score)
    else:  # Black
        return min(1.0, black_score / total_score)


def evaluate_with_neural_net(state, player, eval_model, action_manager, device) -> float:
    """
    Evaluate position using a trained neural network model.
    Neural evaluations are blended with heuristics for robustness.
    """
    try:
        from core.board_encoder import CheckersBoardEncoder
        
        encoder = CheckersBoardEncoder()
        board_state = encoder.encode(state.env.board.get_state(), state.env.current_player)
        state_tensor = board_state.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get Q-values from model
            if hasattr(eval_model, 'get_q_values'):
                q_values = eval_model.get_q_values(state_tensor, player_side=state.env.current_player)
            else:
                q_values = eval_model(state_tensor)
            
            # Use max Q-value as confidence + average as baseline
            max_q = q_values.max().item()
            avg_q = q_values.mean().item()
            
            # Normalize to [0, 1] with emphasis on winning moves (high Q values)
            # Q-values typically in [-1, 1] for well-trained models
            neural_eval = (avg_q + 1.0) / 2.0
            
            # Clamp to valid probability range
            neural_eval = max(0.0, min(1.0, neural_eval))
            
            # Blend with heuristic for safety
            heuristic_eval = evaluate_position(state, player)
            blended = 0.6 * neural_eval + 0.4 * heuristic_eval
            
            return blended
    except Exception as e:
        # Silent fallback - errors in neural eval shouldn't crash MCTS
        return 0.5


def backpropagate(node: MCTSNode, result: float):
    """
    Backpropagate the simulation result up the tree.
    """
    current: Optional[MCTSNode] = node
    while current is not None:
        current.visits += 1
        current.value += result
        
        # Flip result for parent (opponent's perspective)
        result = 1.0 - result
        
        current = current.parent

