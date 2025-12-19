import torch
import random
from typing import Any, Dict, List, Tuple, Optional

from training.d3qn.model import D3QNModel
from training.common.board_encoder import CheckersBoardEncoder
from training.common.action_manager import ActionManager
from training.common.move_parser import parse_legal_moves

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

        action_index = None

        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            # Epsilon-greedy exploration
            if epsilon > 0 and random.random() < epsilon:
                action_index = random.choice(list(mapping.keys()))
            else:
                # Greedy action selection (exploitation)
                mask = self.action_manager.make_legal_action_mask(normalized_moves)
                state = self.encoder.encode(board, player=player, info=info)
                if state.dim() == 3:
                    state = state.unsqueeze(0)
                state = state.to(self.device)

                q_values = self.model.get_q_values(state)

                # Defensive check for unstable model output
                if torch.isnan(q_values).any() or torch.isinf(q_values).any():
                    print("Warning: NaN or Inf detected in Q-values during inference. Falling back to random action.")
                    action_index = random.choice(list(mapping.keys()))
                else:
                    masked = q_values.clone()
                    masked[mask.unsqueeze(0) == 0] = -1e9
                    action_index = int(torch.argmax(masked, dim=1).item())

        # Ensure chosen action_index maps to a legal move
        if action_index is not None and action_index not in mapping:
            action_index = random.choice(list(mapping.keys()))

        if action_index is not None:
            env_move = mapping.get(action_index)
            if env_move is None:
                action_index = random.choice(list(mapping.keys()))
                env_move = mapping[action_index]
            return env_move, action_index
        return None, None
