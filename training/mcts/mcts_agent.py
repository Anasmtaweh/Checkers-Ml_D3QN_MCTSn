import time
from training.mcts.mcts_node import MCTSNode, select, expand, simulate, backpropagate

class MCTSAgent:
    """
    Monte Carlo Tree Search agent for checkers.
    """
    def __init__(self, simulations=1000, time_limit=None, exploration_weight=1.414, eval_model=None, action_manager=None, device=None):
        """
        Args:
            simulations: Number of MCTS simulations to run per move
            time_limit: Time limit in seconds (alternative to simulations)
            exploration_weight: UCB1 exploration parameter (default sqrt(2))
            eval_model: Optional neural network model for position evaluation
            action_manager: ActionManager for encoding actions
            device: Torch device for model evaluation
        """
        self.simulations = simulations
        self.time_limit = time_limit
        self.exploration_weight = exploration_weight
        self.eval_model = eval_model
        self.action_manager = action_manager
        self.device = device
        
        self.nodes_searched = 0
        self.search_time = 0
    
    def get_action(self, state, verbose=False):
        start_time = time.time()
        self.nodes_searched = 0
        
        root = MCTSNode(state, player=state.current_player)
        
        # Time-based with progress
        if self.time_limit:
            print(f"  🤔 MCTS thinking... ", end='', flush=True)
            while time.time() - start_time < self.time_limit:
                self._mcts_iteration(root)
            print(f"✓ ({root.visits} sims in {time.time()-start_time:.1f}s)")
        else:
            # Simulation-based with progress
            for i in range(self.simulations):
                if i % 20 == 0:  # Progress every 20 sims
                    print(f"  {i}/{self.simulations}", end='\r', flush=True)
                self._mcts_iteration(root)
            print()  # New line
        
        self.search_time = time.time() - start_time
        
        # Select best move (highest visit count = most promising)
        if not root.children:
            # No children expanded (shouldn't happen)
            valid_moves = state.get_valid_moves()
            return valid_moves[0] if valid_moves else None
        
        best_child = max(root.children, key=lambda c: c.visits)
        
        if verbose:
            self._print_stats(root, best_child)
        
        return best_child.action
    
    def _mcts_iteration(self, root: MCTSNode):
        """Run one iteration of MCTS (select, expand, simulate, backpropagate)."""
        # 1. Selection
        node = select(root)
        self.nodes_searched += 1
        
        # 2. Expansion
        if not node.is_terminal():
            node = expand(node)
            self.nodes_searched += 1
        
        # 3. Simulation
        result = simulate(node, eval_model=self.eval_model, action_manager=self.action_manager, device=self.device)
        
        # 4. Backpropagation
        backpropagate(node, result)
    
    def _print_stats(self, root: MCTSNode, best_child: MCTSNode):
        """Print search statistics."""
        print(f"\n🔍 MCTS Search Stats:")
        print(f"  Total simulations: {root.visits}")
        print(f"  Nodes searched: {self.nodes_searched}")
        print(f"  Search time: {self.search_time:.2f}s")
        print(f"  Simulations/sec: {root.visits / self.search_time:.0f}")
        print(f"\n🎯 Best move: {best_child.action}")
        print(f"  Visits: {best_child.visits}")
        print(f"  Win rate: {best_child.value / best_child.visits * 100:.1f}%")
        
        # Show top 3 moves
        print(f"\n📊 Top 3 moves:")
        sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)[:3]
        for i, child in enumerate(sorted_children, 1):
            win_rate = (child.value / child.visits * 100) if child.visits > 0 else 0
            print(f"  {i}. {child.action} - Visits: {child.visits}, WR: {win_rate:.1f}%")
