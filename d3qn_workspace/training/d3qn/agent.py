import torch
import random
from typing import Any, Dict, List, Tuple, Optional

from training.d3qn.model import D3QNModel
from core.board_encoder import CheckersBoardEncoder
from core.action_manager import ActionManager
from core.move_parser import parse_legal_moves

Move = Tuple[Tuple[int, int], Tuple[int, int]]


class D3QNAgent:
    """
    Inference-only D3QN agent (no training, no replay buffer).
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

        self.encoder = CheckersBoardEncoder()
        self.action_manager = ActionManager(device=self.device)

        self.model = D3QNModel(
            action_dim=self.action_manager.action_dim,
            device=self.device
        )

    def load_weights(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=True)

        # Case 1: new trainer format: {"online": ..., "target": ...}
        if isinstance(state, dict) and "online" in state and "target" in state:
            self.model.online.load_state_dict(state["online"])
            self.model.target.load_state_dict(state["target"])
            return

        # Case 2: older format: {"model": {"online": ..., "target": ...}}
        if isinstance(state, dict) and "model" in state:
            model_state = state["model"]
            if "online" in model_state and "target" in model_state:
                self.model.online.load_state_dict(model_state["online"])
                self.model.target.load_state_dict(model_state["target"])
            else:
                # Fallback: maybe it's just the online net
                self.model.online.load_state_dict(model_state)
            return

        # Case 3: assume it's directly the online network weights
        self.model.online.load_state_dict(state)

        self.model.online.to(self.device)
        self.model.target.to(self.device)

    # ------------------------------------------------------
    # SELECT ACTION
    # ------------------------------------------------------
    def select_action(self, board, player, legal_moves, epsilon: float = 0.0, info: Optional[Dict] = None) -> Tuple[Optional[Any], Optional[int]]:
        """
        Selects an action using an epsilon-greedy strategy.
        Correctly handles Canonical Perspective (flipping) for Player 2.

        Args:
            board: The current board state (numpy array).
            player: The current player (1 or -1).
            legal_moves: A list of legal moves from the environment.
            epsilon: The probability of choosing a random action.
            info: Optional info dict from the environment.

        Returns:
            A tuple of (environment_move, action_index).
            Returns (None, None) if no legal moves are available.
        """
        normalized_moves, mapping = parse_legal_moves(legal_moves, self.action_manager)
        if not mapping:
            return None, None

        # Prepare Canonical Mask & Mapping
        # If Player 2, the network sees a flipped board (P1 perspective).
        # So we must flip the legal moves to match what the network expects to output.
        if player == -1:
            canonical_moves = [self.action_manager.flip_move(m) for m in normalized_moves]
            mask = self.action_manager.make_legal_action_mask(canonical_moves)
            
            # Map Canonical ID -> Absolute ID
            canonical_to_absolute = {}
            for i, cm in enumerate(canonical_moves):
                cid = self.action_manager.get_action_id(cm)
                if cid >= 0:
                    orig_move = normalized_moves[i]
                    aid = self.action_manager.get_action_id(orig_move)
                    canonical_to_absolute[cid] = aid
        else:
            # Player 1: Canonical = Absolute
            mask = self.action_manager.make_legal_action_mask(normalized_moves)
            canonical_to_absolute = None

        action_index = None

        self.model.eval()
        with torch.no_grad():
            # Epsilon-greedy exploration
            if epsilon > 0 and random.random() < epsilon:
                absolute_id = random.choice(list(mapping.keys()))
                return mapping[absolute_id], absolute_id
            else:
                # Greedy action selection (exploitation)
                state = self.encoder.encode(board, player=player, info=info)
                if state.dim() == 3:
                    state = state.unsqueeze(0)
                state = state.to(self.device)

                q_values = self.model.get_q_values(state)

                # Defensive check for unstable model output
                if torch.isnan(q_values).any() or torch.isinf(q_values).any():
                    print("Warning: NaN or Inf detected in Q-values during inference. Falling back to random action.")
                    absolute_id = random.choice(list(mapping.keys()))
                    return mapping[absolute_id], absolute_id
                else:
                    masked = q_values.clone()
                    masked[mask.unsqueeze(0) == 0] = -1e9
                    action_index = int(torch.argmax(masked, dim=1).item())

        # Convert Canonical Action Index -> Absolute Action Index -> Env Move
        absolute_id = None
        if action_index is not None:
            if player == -1 and canonical_to_absolute is not None:
                absolute_id = canonical_to_absolute.get(action_index)
            else:
                absolute_id = action_index

        if absolute_id is not None and absolute_id in mapping:
            return mapping[absolute_id], absolute_id
            
        # Fallback
        absolute_id = random.choice(list(mapping.keys()))
        return mapping[absolute_id], absolute_id
