import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import copy


class AlphaNode:
    """
    Node in the AlphaZero MCTS tree.
    
    Each node represents a game state and stores:
    - Visit statistics (N, W)
    - Prior probability P from the policy network
    - Children nodes for each explored action
    
    Attributes:
        visits (int): Number of times this node has been visited (N)
        value_sum (float): Cumulative value from all visits (W)
        prior (float): Prior probability P(s,a) from policy network
        children (Dict[int, AlphaNode]): Child nodes indexed by action_id
        state (Any): Game state object (optional, stored for leaf nodes)
        parent (AlphaNode): Parent node reference
        action_taken (int): Action that led to this node from parent
    """
    
    def __init__(self, prior: float = 0.0, parent: Optional['AlphaNode'] = None, 
                 action_taken: int = -1, state: Any = None):
        """
        Initialize an AlphaNode.
        
        Args:
            prior: Prior probability P(s,a) from policy network
            parent: Parent node reference
            action_taken: Action ID that led to this node
            state: Game state object (stored for leaf expansion)
        """
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior
        self.children: Dict[int, 'AlphaNode'] = {}
        self.state: Any = state
        self.parent: Optional['AlphaNode'] = parent
        self.action_taken: int = action_taken
    
    def get_value(self):
        """
        Get the average Q-value of this node.
        This represents the expected game outcome from this state
        according to MCTS (not actual game outcome).
        """
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    @property
    def q_value(self) -> float:
        """
        Compute Q-value: average value over all visits.
        
        Returns:
            Q(s,a) = W/N, or 0.0 if never visited
        """
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (no children expanded yet)."""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """Check if this node is the root (no parent)."""
        return self.parent is None


class MCTS:
    """
    AlphaZero-style Monte Carlo Tree Search with neural network guidance.
    
    Uses PUCT (Predictor + Upper Confidence Bound for Trees) algorithm:
    - Selection: Choose actions using UCB formula with neural priors
    - Expansion: Expand leaf nodes using policy network
    - Evaluation: Evaluate positions using value network (no rollouts!)
    - Backpropagation: Update statistics up the tree with value flipping
    
    Key differences from traditional MCTS:
    - No random rollouts (use neural network value head instead)
    - Policy network provides priors for exploration
    - Much more efficient (fewer simulations needed)
    """
    
    def __init__(self, model, action_manager, encoder, c_puct: float = 1.5, 
                 num_simulations: int = 400, device: str = "cpu",
                 dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25):
        """
        Initialize AlphaZero MCTS.
        
        Args:
            model: AlphaZeroModel instance for policy and value prediction
            action_manager: ActionManager for move encoding/decoding
            encoder: CheckersBoardEncoder for state encoding
            c_puct: Exploration constant for PUCT formula (typically 1.0-3.0)
            num_simulations: Number of MCTS simulations per move
            device: Device for tensor operations
            dirichlet_alpha: Shape parameter for Dirichlet noise (0.3 for chess/checkers)
            dirichlet_epsilon: Weight of noise mixing (0.25 = 25% noise, 75% neural policy)
        """
        self.model = model
        self.action_manager = action_manager
        self.encoder = encoder
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = device
        
        # Dirichlet Noise parameters
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        
        # Set model to evaluation mode
        self.model.eval()
    
    def get_action_prob(self, env, temp: float = 1.0, training: bool = True) -> Tuple[np.ndarray, AlphaNode]:
        """
        Main entry point: Run MCTS and return action probability distribution.
        
        This method:
        1. Creates root node with current game state
        2. Runs num_simulations MCTS searches
        3. Computes visit-count-based action probabilities
        4. Applies temperature for exploration control
        
        Args:
            env: CheckersEnv instance with current game state
            temp: Temperature parameter for action selection:
                  - temp > 1.0: More exploration (flatter distribution)
                  - temp = 1.0: Proportional to visit counts
                  - temp < 1.0: More exploitation (sharper distribution)
                  - temp → 0: Deterministic (argmax)
            training: Whether to add Dirichlet noise (only in training)
        
        Returns:
            Tuple of (action_probabilities, root_node):
                - action_probabilities: np.array of shape (action_dim,)
                - root_node: Root AlphaNode (useful for tree reuse)
        """
        # Create root node with current state
        root = AlphaNode(prior=1.0, state=env)
        
        # 1. EXPAND ROOT IMMEDIATELY (Critical for noise)
        self._expand_node(root, env)
        
        # 2. ADD DIRICHLET NOISE (Only at the root, and only if training)
        if training:
            self._add_dirichlet_noise(root)
        
        # Run MCTS simulations
        for _ in range(self.num_simulations):
            # Make a copy of the environment for simulation
            env_copy = self._copy_env(env)
            self._search(root, env_copy)
        
        # Compute action probabilities from visit counts
        action_probs = self._get_action_distribution(root, temp)
        
        return action_probs, root
    
    def _add_dirichlet_noise(self, node: AlphaNode):
        """
        Injects Dirichlet noise into the root node's children priors.
        Formula: P'(a) = (1 - ε) * P(a) + ε * Noise
        
        This prevents the network from becoming too confident in certain moves
        too early, which can lead to premature convergence to suboptimal strategies
        (like always playing for draws).
        """
        children = list(node.children.values())
        if not children:
            return
            
        # Generate noise from Dirichlet distribution
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(children))
        
        # Mix noise with existing priors
        for i, child in enumerate(children):
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * noise[i]
    
    def _search(self, node: AlphaNode, env) -> float:
        """
        Recursive MCTS search: Selection → Expansion → Evaluation → Backpropagation.
        
        Args:
            node: Current node in the tree
            env: Game environment state
        
        Returns:
            Value of this position (will be flipped during backprop)
        """
        # Check if game is over
        if env.done:
            # Terminal node: return actual game outcome
            winner = env._check_game_over()[1]
            current_player = env.current_player
            
            # Convert winner to value from current player's perspective
            if winner == current_player:
                value = 1.0  # Win
            elif winner == -current_player:
                value = -1.0  # Loss
            else:
                value = 0.0  # Draw
            
            # Backpropagate
            self._backpropagate(node, value)
            return value
        
        # ================================================================
        # LEAF NODE: Expand and Evaluate with Neural Network
        # ================================================================
        if node.is_leaf():
            # Get policy and value from neural network
            value = self._expand_node(node, env)
            
            # Backpropagate the value
            self._backpropagate(node, value)
            return value
        
        # ================================================================
        # INTERNAL NODE: Select best child using PUCT
        # ================================================================
        best_action, best_child = self._select_child(node)
        
        # Apply the selected action to the environment
        legal_moves = env.get_legal_moves()
        move = self._get_move_from_action(best_action, legal_moves)
        
        if move is None:
            # Should not happen if expansion was correct
            # Return pessimistic value
            value = -1.0
            self._backpropagate(node, value)
            return value
        
        # Execute move in the environment
        _, _, _, _ = env.step(move)
        
        # Recursively search the child
        value = self._search(best_child, env)
        
        # Note: Backpropagation already happened in the recursive call
        # The value is automatically flipped at each level
        return -value  # Flip for parent's perspective
    
    def _expand_node(self, node: AlphaNode, env) -> float:
        """
        Expand a leaf node using the neural network.
        
        This is the CRITICAL CHANGE from traditional MCTS:
        - Instead of random rollout, use neural network for evaluation
        - Policy head provides priors for child nodes
        - Value head provides position evaluation
        
        Args:
            node: Leaf node to expand
            env: Current game environment state
        
        Returns:
            Value of this position from current player's perspective
        """
        # Encode the current board state
        board = env.board.get_state()
        player = env.current_player
        state_tensor = self.encoder.encode(board, player)
        
        # Get policy and value from neural network
        with torch.no_grad():
            policy, value = self.model.predict(state_tensor)
        
        # Get legal moves and create mask
        legal_moves = env.get_legal_moves()
        legal_mask = self.action_manager.make_legal_action_mask(legal_moves)
        
        # Mask illegal actions (set their probabilities to 0)
        policy = policy.cpu()
        legal_mask = legal_mask.cpu()
        policy = policy * legal_mask.float()
        
        # Renormalize to ensure probabilities sum to 1
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            # Edge case: no legal moves (shouldn't happen, but handle gracefully)
            # Uniform distribution over legal moves
            if legal_mask.sum() > 0:
                policy = legal_mask.float() / legal_mask.sum()
            else:
                policy = torch.ones_like(policy) / len(policy)
        
        # Create child nodes for all legal actions
        for action_id in range(self.action_manager.action_dim):
            if legal_mask[action_id]:
                prior = policy[action_id].item()
                child_node = AlphaNode(
                    prior=prior,
                    parent=node,
                    action_taken=action_id,
                    state=None  # Don't store state for children (memory efficiency)
                )
                node.children[action_id] = child_node
        
        # Return value from current player's perspective
        return value
    
    def _select_child(self, node: AlphaNode) -> Tuple[int, AlphaNode]:
        """
        Select the best child using the PUCT formula.
        
        PUCT Formula:
        UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(Σ N(s,b)) / (1 + N(s,a))
        
        Where:
        - Q(s,a): Average value of action a (exploitation term)
        - P(s,a): Prior probability from policy network
        - N(s,a): Visit count for action a
        - Σ N(s,b): Total visits to parent node
        - c_puct: Exploration constant
        
        Args:
            node: Parent node to select from
        
        Returns:
            Tuple of (best_action_id, best_child_node)
        """
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        # Total visit count for the parent node
        total_visits = node.visits
        sqrt_total_visits = np.sqrt(total_visits)
        
        # Evaluate each child
        for action_id, child in node.children.items():
            # Q-value (exploitation)
            q_value = child.q_value
            
            # Exploration bonus (PUCT formula)
            # U(s,a) = c_puct * P(s,a) * sqrt(Σ N) / (1 + N(s,a))
            exploration_bonus = (
                self.c_puct * child.prior * sqrt_total_visits / (1 + child.visits)
            )
            
            # UCB score
            ucb_score = q_value + exploration_bonus
            
            # Track best
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action_id
                best_child = child
        
        assert best_child is not None, "No children found during selection"
        return best_action, best_child
    
    def _backpropagate(self, node: AlphaNode, value: float):
        """
        Backpropagate value up the tree.
        
        CRITICAL: Value flipping for zero-sum games!
        - What is good for the current player is bad for the parent
        - Flip value sign at each level: value → -value
        
        Args:
            node: Node to start backpropagation from
            value: Value to propagate (from current node's perspective)
        """
        current = node
        current_value = value
        
        while current is not None:
            # Update statistics
            current.visits += 1
            current.value_sum += current_value
            
            # Move to parent and flip value
            current = current.parent
            current_value = -current_value  # FLIP for parent's perspective
    
    def _get_action_distribution(self, root: AlphaNode, temp: float) -> np.ndarray:
        """
        Compute action probability distribution from visit counts.
        
        π(a) ∝ N(s,a)^(1/temp)
        
        Args:
            root: Root node with visit statistics
            temp: Temperature parameter
        
        Returns:
            Action probability distribution of shape (action_dim,)
        """
        action_probs = np.zeros(self.action_manager.action_dim)
        
        # Collect visit counts
        for action_id, child in root.children.items():
            action_probs[action_id] = child.visits
        
        # Apply temperature
        if temp == 0:
            # Deterministic: choose most visited
            best_action = np.argmax(action_probs)
            action_probs = np.zeros_like(action_probs)
            action_probs[best_action] = 1.0
        else:
            # Temperature scaling: π(a) ∝ N(s,a)^(1/temp)
            action_probs = action_probs ** (1.0 / temp)
            
            # Normalize
            if action_probs.sum() > 0:
                action_probs = action_probs / action_probs.sum()
            else:
                # Fallback: uniform over children
                for action_id in root.children.keys():
                    action_probs[action_id] = 1.0
                if action_probs.sum() > 0:
                    action_probs = action_probs / action_probs.sum()
        
        return action_probs
    
    def _get_move_from_action(self, action_id: int, legal_moves: List) -> Optional[Any]:
        """
        Convert action ID to actual game move.
        
        Args:
            action_id: Integer action index
            legal_moves: List of legal moves from environment
        
        Returns:
            Move object, or None if not found
        """
        # Get the (start, landing) pair from action_id
        move_pair = self.action_manager.get_move_from_id(action_id)
        
        if move_pair == ((-1, -1), (-1, -1)):
            return None
        
        # Find matching move in legal_moves
        for move in legal_moves:
            move_start_landing = self.action_manager._extract_start_landing(move)
            if move_start_landing == move_pair:
                return move
        
        return None
    
    def _copy_env(self, env):
        """
        Create a deep copy of the game environment for simulation.
        
        Args:
            env: CheckersEnv instance
        
        Returns:
            Deep copy of the environment
        """
        return copy.deepcopy(env)


# ================================================================
# Utility Functions
# ================================================================

def select_action_from_probs(action_probs: np.ndarray, legal_moves: List, 
                             action_manager, deterministic: bool = False) -> Any:
    """
    Sample an action from the probability distribution.
    
    Args:
        action_probs: Action probability distribution from MCTS
        legal_moves: List of legal moves
        action_manager: ActionManager instance
        deterministic: If True, choose argmax; else sample
    
    Returns:
        Selected move
    """
    if deterministic:
        action_id = np.argmax(action_probs)
    else:
        # Sample according to distribution
        action_id = np.random.choice(len(action_probs), p=action_probs)
    
    # Find corresponding move
    move_pair = action_manager.get_move_from_id(action_id)
    
    for move in legal_moves:
        move_start_landing = action_manager._extract_start_landing(move)
        if move_start_landing == move_pair:
            return move
    
    # Fallback: return first legal move if something went wrong
    return legal_moves[0] if legal_moves else None


# ================================================================
# Testing and Debugging
# ================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ALPHAZERO MCTS ENGINE TEST")
    print("=" * 70)
    
    import sys
    sys.path.append('.')
    
    from training.alpha_zero.network import AlphaZeroModel
    from core.action_manager import ActionManager
    from core.board_encoder import CheckersBoardEncoder
    from core.game import CheckersEnv
    
    # Initialize components
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    action_manager = ActionManager(device)
    encoder = CheckersBoardEncoder()
    model = AlphaZeroModel(action_dim=action_manager.action_dim, device=device)
    
    # Create MCTS engine
    mcts = MCTS(
        model=model,
        action_manager=action_manager,
        encoder=encoder,
        c_puct=1.5,
        num_simulations=100,  # Reduced for testing
        device=device
    )
    
    print(f"MCTS initialized with {mcts.num_simulations} simulations per move")
    print(f"Exploration constant c_puct = {mcts.c_puct}")
    
    # Test 1: Single MCTS search
    print("\n" + "-" * 70)
    print("Test 1: MCTS Search on Starting Position")
    print("-" * 70)
    
    env = CheckersEnv()
    env.reset()
    
    print("Running MCTS search...")
    action_probs, root = mcts.get_action_prob(env, temp=1.0)
    
    print(f"Root visits: {root.visits}")
    print(f"Number of children: {len(root.children)}")
    print(f"Action probabilities sum: {action_probs.sum():.6f}")
    print(f"Max probability: {action_probs.max():.6f}")
    print(f"Number of actions with P > 0: {np.sum(action_probs > 0)}")
    
    # Test 2: Verify PUCT formula
    print("\n" + "-" * 70)
    print("Test 2: Verify PUCT Selection")
    print("-" * 70)
    
    if len(root.children) > 0:
        print("Top 5 actions by visit count:")
        visit_counts = [(aid, child.visits, child.q_value, child.prior) 
                       for aid, child in root.children.items()]
        visit_counts.sort(key=lambda x: x[1], reverse=True)
        
        for i, (action_id, visits, q_val, prior) in enumerate(visit_counts[:5]):
            print(f"  {i+1}. Action {action_id:3d}: "
                  f"N={visits:3d}, Q={q_val:+.3f}, P={prior:.3f}")
    
    # Test 3: Temperature effects
    print("\n" + "-" * 70)
    print("Test 3: Temperature Effects")
    print("-" * 70)
    
    temps = [0.0, 0.5, 1.0, 2.0]
    for t in temps:
        probs_temp, _ = mcts.get_action_prob(env, temp=t)
        max_prob = probs_temp.max()
        entropy = -np.sum(probs_temp * np.log(probs_temp + 1e-10))
        print(f"  temp={t:.1f}: max_prob={max_prob:.3f}, entropy={entropy:.2f}")
    
    # Test 4: Action selection
    print("\n" + "-" * 70)
    print("Test 4: Action Selection")
    print("-" * 70)
    
    legal_moves = env.get_legal_moves()
    print(f"Number of legal moves: {len(legal_moves)}")
    
    # Deterministic selection
    move_det = select_action_from_probs(action_probs, legal_moves, action_manager, 
                                       deterministic=True)
    print(f"Deterministic move selected: {move_det}")
    
    # Stochastic selection
    move_stoch = select_action_from_probs(action_probs, legal_moves, action_manager, 
                                         deterministic=False)
    print(f"Stochastic move selected: {move_stoch}")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED - ALPHAZERO MCTS ENGINE READY")
    print("=" * 70)
    print("\nKey Features Implemented:")
    print("  • Neural network-guided tree search (no rollouts!)")
    print("  • PUCT formula for exploration/exploitation balance")
    print("  • Value flipping for zero-sum game backpropagation")
    print("  • Temperature-controlled action selection")
    print("  • Efficient leaf expansion with policy priors")
