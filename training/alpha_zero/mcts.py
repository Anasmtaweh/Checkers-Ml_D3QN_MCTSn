import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import copy

class AlphaNode:
    """Node in the AlphaZero MCTS tree."""
    def __init__(self, prior: float = 0.0, parent: Optional['AlphaNode'] = None, 
                 action_taken: int = -1, state: Any = None, player_to_move: int = 1):
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior
        self.children: Dict[int, 'AlphaNode'] = {}
        self.state: Any = state
        self.parent: Optional['AlphaNode'] = parent
        self.action_taken: int = action_taken
        self.player_to_move: int = player_to_move
    
    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_greedy_value(self) -> float:
        return self.q_value

class MCTS:
    """AlphaZero MCTS with Fixed Value Flipping."""
    
    def __init__(self, model, action_manager, encoder, c_puct: float = 1.5, 
                 num_simulations: int = 400, device: str = "cpu",
                 dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25):
        self.model = model
        self.action_manager = action_manager
        self.encoder = encoder
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = device
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.model.eval()
    
    def get_action_prob(self, env, temp: float = 1.0, training: bool = True) -> Tuple[np.ndarray, AlphaNode]:        # Create root
        root = AlphaNode(prior=1.0, state=env, player_to_move=env.current_player)
        
        # 1. Expand Root
        self._expand_node(root, env)
        
        # 2. Add Noise (Training only)
        if training and temp > 0:
            self._add_dirichlet_noise(root)
        
        # 3. Simulations
        for _ in range(self.num_simulations):
            env_copy = self._copy_env(env)
            self._search(root, env_copy)
        
        # 4. Action Distribution
        action_probs = self._get_action_distribution(root, temp)
        return action_probs, root
    
    def _add_dirichlet_noise(self, node: AlphaNode):
        children = list(node.children.values())
        if not children: return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(children))
        for i, child in enumerate(children):
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * noise[i]
    
    def _search(self, node: AlphaNode, env) -> float:
        # Terminal check
        if env.done:
            # Terminal evaluation from node's perspective
            # draw -> 0.0
            # else -> +1.0 if winner == node.player_to_move else -1.0
            if env.winner == 0:
                value = -0.1  # Slight negative for draw to prefer winning lines
            else:
                value = 1.0 if env.winner == node.player_to_move else -1.0
            
            # Update stats locally (Recursive Backprop)
            node.value_sum += value
            node.visits += 1
            return value
        
        # Leaf -> Expand
        if node.is_leaf():
            value = self._expand_node(node, env)
            node.value_sum += value
            node.visits += 1
            return value
        
        # Internal -> Select
        best_action, best_child = self._select_child(node)
        
        # Move
        legal_moves = env.get_legal_moves()
        move = self._get_move_from_action(best_action, legal_moves, env.current_player)
        
        if move is None:
            # Fallback for rare edge cases (e.g. no legal moves but not done)
            value = -1.0
            node.value_sum += value
            node.visits += 1
            return value
            
        env.step(move)
        
        # Update child's player_to_move after step
        if best_child is None:
            raise RuntimeError("MCTS: Selected child is None for internal node")
        best_child.player_to_move = env.current_player
        
        # Recurse and FLIP value (The Single Flipping Mechanism)
        value = -self._search(best_child, env)
        
        # Update stats locally
        node.value_sum += value
        node.visits += 1
        return value
    
    def _expand_node(self, node: AlphaNode, env) -> float:
        board = env.board.get_state()
        player = env.current_player
        state_tensor = self.encoder.encode(board, player)
        
        with torch.no_grad():
            policy, value = self.model.predict(state_tensor)
        
        # Ensure value is a Python float
        value = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
        
        legal_moves = env.get_legal_moves()

        # IMPORTANT: pass player for canonical masking
        legal_mask = self.action_manager.make_legal_action_mask(legal_moves, player=player)
        
        # Work on CPU for stable behavior
        legal_mask_cpu = legal_mask.detach().cpu()
        policy_cpu = policy.detach().cpu()

        # Mask illegal actions
        policy_cpu = policy_cpu * legal_mask_cpu.float()

        # Normalize (fallback to uniform over legal if network gives all-zero)
        if policy_cpu.sum() > 0:
            policy_cpu = policy_cpu / policy_cpu.sum()
        else:
            policy_cpu = legal_mask_cpu.float() / (legal_mask_cpu.sum() + 1e-8)
        
        for action_id in range(self.action_manager.action_dim):
            if bool(legal_mask_cpu[action_id]):
                # Child's player_to_move will be set in _search after env.step()
                child = AlphaNode(prior=float(policy_cpu[action_id].item()), parent=node, action_taken=action_id)
                node.children[action_id] = child
        
        return value
    
    def _select_child(self, node: AlphaNode) -> Tuple[int, Optional[AlphaNode]]:
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        # Ensure exploration works from the first visit
        sqrt_total_visits = np.sqrt(node.visits + 1)
        
        for action_id, child in node.children.items():
            # Value is flipped because child stores value for opponent
            q_value = -child.q_value  
            
            exploration_bonus = self.c_puct * child.prior * sqrt_total_visits / (1 + child.visits)
            ucb_score = q_value + exploration_bonus
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action_id
                best_child = child
        
        return best_action, best_child
    
    def _get_action_distribution(self, root: AlphaNode, temp: float) -> np.ndarray:
        probs = np.zeros(self.action_manager.action_dim)
        for aid, child in root.children.items():
            probs[aid] = child.visits
        
        if temp == 0:
            best = np.argmax(probs)
            probs = np.zeros_like(probs)
            probs[best] = 1.0
        else:
            probs = probs ** (1.0 / temp)
            if probs.sum() > 0: probs /= probs.sum()
        return probs

    def _get_move_from_action(self, action_id: int, legal_moves: List, player: int = 1) -> Optional[Any]:
        move_pair = self.action_manager.get_move_from_id(action_id)
        if move_pair == ((-1, -1), (-1, -1)): return None
        
        # If player is -1, the network's canonical move must be flipped to match real board coordinates
        if player == -1:
            move_pair = self.action_manager.flip_move(move_pair)
            
        for move in legal_moves:
            if self.action_manager._extract_start_landing(move) == move_pair:
                return move
        return None

    def _copy_env(self, env):
        return copy.deepcopy(env)