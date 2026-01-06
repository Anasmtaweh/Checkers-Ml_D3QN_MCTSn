import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from core.game import CheckersEnv

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
        self.is_forced: bool = False

    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def get_greedy_value(self) -> float:
        return self.q_value

    def is_leaf(self) -> bool:
        return len(self.children) == 0

class MCTS:
    """AlphaZero MCTS optimized for Checkers performance and 6-channel context."""
    
    def __init__(self, model, action_manager, encoder, c_puct: float = 1.5, 
                 num_simulations: int = 400, device: str = "cpu",
                 dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25,
                 draw_value: float = 0.0, search_draw_bias: float = -0.02,
                 skip_root_sims_on_forced: bool = True):
        self.model = model
        self.action_manager = action_manager
        self.encoder = encoder
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = device
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.draw_value = float(draw_value)
        self.search_draw_bias = float(search_draw_bias)
        self.skip_root_sims_on_forced = skip_root_sims_on_forced
        self.model.eval()
    
    def get_action_prob(self, env, temp: float = 1.0, training: bool = True) -> Tuple[np.ndarray, AlphaNode]:
        # --- ENDGAME DETECTOR (Robust Version) ---
        # Count pieces for both sides (Handles Kings 2/-2 and Men 1/-1)
        board = env.board.board
        red_count = np.count_nonzero(board > 0)   # All Red pieces
        black_count = np.count_nonzero(board < 0) # All Black pieces
        
        total_pieces = red_count + black_count
        
        # Trigger High Exploration if:
        # 1. The board is mostly empty (<= 10 pieces)
        # 2. OR One side is down to a few pieces (<= 6), creating a "Conversion Phase"
        self.is_endgame = (
            total_pieces <= 10 
            or min(red_count, black_count) <= 6
        )

        # SPEED OPTIMIZATION: Short-circuit for mandatory moves
        legal_moves = env.get_legal_moves()
        if self.skip_root_sims_on_forced and len(legal_moves) == 1:
            move = legal_moves[0]
            sl = self.action_manager._extract_start_landing(move)
            if env.current_player == -1: sl = self.action_manager.flip_move(sl)
            action_id = self.action_manager.get_action_id(sl)
            
            probs = np.zeros(self.action_manager.action_dim, dtype=np.float32)
            probs[action_id] = 1.0
            
            root = AlphaNode(prior=1.0, state=env, player_to_move=env.current_player)
            root.children[action_id] = AlphaNode(prior=1.0, parent=root, action_taken=action_id)
            return probs, root

        root = AlphaNode(prior=1.0, state=env, player_to_move=env.current_player)
        self._expand_node(root, env)

        if training and temp > 0 and len(root.children) > 1:
            self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            env_copy = self._copy_env(env)
            self._search(root, env_copy)

        return self._get_action_distribution(root, temp), root
    
    def _add_dirichlet_noise(self, node: AlphaNode):
        children = list(node.children.values())
        if not children: return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(children))
        for i, child in enumerate(children):
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * noise[i]
    
    def _search(self, node: AlphaNode, env: CheckersEnv) -> float:
        if env.done:
            if env.winner == 0:
                value = self.draw_value
            else:
                value = 1.0 if env.winner == node.player_to_move else -1.0
            node.value_sum += value
            node.visits += 1
            return value

        if node.is_leaf():
            value = self._expand_node(node, env)
            node.value_sum += value
            node.visits += 1
            return value

        # Select
        if len(node.children) == 1:
            best_action, best_child = next(iter(node.children.items()))
        else:
            best_action, best_child = self._select_child(node)
        
        if best_child is None:
            raise RuntimeError("MCTS selection failed: Node has children but no child selected.")

        player_before = env.current_player
        move = self._get_move_from_action(best_action, env.get_legal_moves(), env.current_player)
        
        if move is None:
            # Penalize the child so we don't pick it again
            best_child.player_to_move = node.player_to_move
            best_child.value_sum += -1.0
            best_child.visits += 1
            
            v = -1.0
            node.value_sum += v; node.visits += 1; return v
            
        env.step(move)
        best_child.player_to_move = env.current_player
        
        # Turn-Aware Recursive Backprop
        if not env.done and env.current_player == player_before:
            value = self._search(best_child, env)
        else:
            value = -self._search(best_child, env)
        
        node.value_sum += value
        node.visits += 1
        return value
    
    def _expand_node(self, node: AlphaNode, env: CheckersEnv) -> float:
        board = env.board.get_state()
        player = env.current_player
        
        # CONTEXT FIX: Pass force_capture_from to the encoder for the 6th channel
        state_tensor = self.encoder.encode(board, player, force_move_from=env.force_capture_from)
        
        with torch.no_grad():
            policy, value = self.model.predict(state_tensor)
        
        value = float(value)
        legal_moves = env.get_legal_moves()
        node.is_forced = (len(legal_moves) == 1)
        
        legal_mask = self.action_manager.make_legal_action_mask(legal_moves, player=player)
        legal_mask_cpu = legal_mask.detach().cpu()
        p_cpu = policy.detach().cpu() * legal_mask_cpu.float()

        if p_cpu.sum() > 0: p_cpu /= p_cpu.sum()
        else: p_cpu = legal_mask_cpu.float() / (legal_mask_cpu.sum() + 1e-8)
        
        for action_id in range(self.action_manager.action_dim):
            if bool(legal_mask_cpu[action_id]):
                child = AlphaNode(prior=float(p_cpu[action_id].item()), parent=node, action_taken=action_id)
                node.children[action_id] = child
        return value
    
    def _select_child(self, node: AlphaNode) -> Tuple[int, Optional[AlphaNode]]:
        best_score = -float('inf'); best_action = -1; best_child = None
        sqrt_total = np.sqrt(node.visits + 1)
        
        # DYNAMIC PUCT SCALING
        # If in endgame mode, double the exploration to break shuffling loops.
        # Otherwise use standard c_puct.
        current_cpuct = self.c_puct * 2.0 if getattr(self, 'is_endgame', False) else self.c_puct

        for action_id, child in node.children.items():
            if child.visits > 0 and child.player_to_move == node.player_to_move:
                q = child.q_value
            else:
                q = -child.q_value  
            
            # Anti-Lazy Bias
            if abs(q) < 0.1: q += self.search_draw_bias
            
            # Use the dynamic cpuct here
            score = q + current_cpuct * child.prior * sqrt_total / (1 + child.visits)
            
            if score > best_score:
                best_score = score; best_action = action_id; best_child = child
        return best_action, best_child

    def _get_action_distribution(self, root: AlphaNode, temp: float) -> np.ndarray:
        probs = np.zeros(self.action_manager.action_dim, dtype=np.float32)
        for aid, child in root.children.items(): probs[aid] = child.visits
        if temp == 0:
            best = int(np.argmax(probs))
            onehot = np.zeros_like(probs); onehot[best] = 1.0
            return onehot
        probs = probs ** (1.0 / temp)
        if probs.sum() > 0: probs /= probs.sum()
        return probs

    def _get_move_from_action(self, action_id: int, legal_moves: List, player: int = 1) -> Optional[Any]:
        mp = self.action_manager.get_move_from_id(action_id)
        if mp == ((-1, -1), (-1, -1)): return None
        if player == -1: mp = self.action_manager.flip_move(mp)
        for move in legal_moves:
            if self.action_manager._extract_start_landing(move) == mp: return move
        return None

    def _copy_env(self, env: CheckersEnv) -> CheckersEnv:
        """Lightweight manual copy to replace slow deepcopy."""
        new_env = CheckersEnv(max_moves=env.max_moves, no_progress_limit=env.no_progress_limit)
        new_env.board.board = env.board.get_state() # numpy copy
        new_env.current_player = env.current_player
        new_env.force_capture_from = env.force_capture_from
        new_env.move_count = env.move_count
        new_env.no_progress = env.no_progress
        new_env.done = env.done
        new_env.winner = env.winner
        return new_env