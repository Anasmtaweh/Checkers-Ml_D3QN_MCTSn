import torch
from typing import Any, Dict, List, Tuple, Optional

from training.d3qn.model import D3QNModel
from training.common.board_encoder import CheckersBoardEncoder
from training.common.action_manager import ActionManager

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
    # LEGAL MOVE NORMALIZATION (mirrors trainer behavior)
    # ------------------------------------------------------
    def _normalize_move(self, mv):
        """
        Normalizes all move formats produced by CheckersEnv into a
        sequence of (row, col) pairs: [(r1,c1), (r2,c2), ...].

        Handles:
        - ((r1,c1),(r2,c2))
        - ((r1,c1),(r_mid,c_mid),(r2,c2))       # capture triple
        - [( (r1,c1),(r2,c2) )]
        - [( (r1,c1),(r_mid,c_mid),(r2,c2) )]
        - [(r1,c1),(r2,c2),(r3,c3)]             # multi-jumps
        """
        # Unwrap list-wrapped moves: [((...),(...))]
        if isinstance(mv, list) and len(mv) == 1 and isinstance(mv[0], (list, tuple)):
            mv = mv[0]

        # Now mv should be a tuple/list of steps
        if not isinstance(mv, (list, tuple)):
            return None

        steps = []

        for s in mv:
            # s must be (r,c)
            if isinstance(s, (list, tuple)) and len(s) == 2:
                r, c = s
                if isinstance(r, int) and isinstance(c, int):
                    steps.append((r, c))
                else:
                    # If r or c is not int → invalid format
                    return None
            else:
                # Unknown shape
                return None

        return steps

    def _prepare_legal(self, legal_moves_raw: List[Any]) -> Tuple[List[Move], Dict[int, Any]]:
        """
        Mirror the trainer's _build_legal_from_raw:

        - Input: env-format legal moves (pairs, triples, or sequences of triples)
        - Output:
            normalized_moves: list of ((r1,c1), (r2,c2)) for masking/indexing
            mapping: action_index -> env_action (pair or single capture triple)
        """
        normalized_moves: List[Move] = []
        mapping: Dict[int, Any] = {}
        has_capture = False
        candidates: List[Tuple[Any, Move, bool]] = []

        for mv in legal_moves_raw:
            # We only handle list/tuple moves
            if not isinstance(mv, (list, tuple)):
                continue

            env_action: Optional[Any] = None
            norm: Optional[Move] = None
            is_capture = False

            # Case 1: multi-step capture sequence: [ (from, to, jumped), ... ]
            if (
                isinstance(mv, (list, tuple))
                and len(mv) > 0
                and isinstance(mv[0], (list, tuple))
                and len(mv[0]) == 3
            ):
                step0 = mv[0]
                if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in step0):
                    (r1, c1), (r2, c2), jumped = step0
                    r1, c1 = int(r1), int(c1)
                    r2, c2 = int(r2), int(c2)
                    jr, jc = int(jumped[0]), int(jumped[1])
                    norm = ((r1, c1), (r2, c2))
                    env_action = ((r1, c1), (r2, c2), (jr, jc))
                    is_capture = True

            # Case 2: single capture step triple: (from, to, jumped)
            elif (
                len(mv) == 3
                and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in mv)
            ):
                (r1, c1), (r2, c2), jumped = mv
                r1, c1 = int(r1), int(c1)
                r2, c2 = int(r2), int(c2)
                jr, jc = int(jumped[0]), int(jumped[1])
                norm = ((r1, c1), (r2, c2))
                env_action = ((r1, c1), (r2, c2), (jr, jc))
                is_capture = True

            # Case 3: simple quiet move: ((r1,c1), (r2,c2))
            elif (
                len(mv) == 2
                and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in mv)
            ):
                (r1, c1), (r2, c2) = mv
                r1, c1 = int(r1), int(c1)
                r2, c2 = int(r2), int(c2)
                norm = ((r1, c1), (r2, c2))
                env_action = ((r1, c1), (r2, c2))
                # is_capture stays False

            # Skip if we couldn't parse the move
            if norm is None or env_action is None:
                continue

            candidates.append((env_action, norm, is_capture))
            if is_capture:
                has_capture = True

        # Second pass: enforce mandatory capture & build index->env_action/mask moves
        for env_action, norm, is_capture in candidates:
            if has_capture and not is_capture:
                # quiet moves illegal if any capture exists
                continue
            try:
                idx = self.action_manager.encode_move(norm)
            except ValueError:
                # move outside action space, ignore
                continue
            mapping[idx] = env_action
            normalized_moves.append(norm)

        return normalized_moves, mapping

    # ------------------------------------------------------
    # SELECT ACTION
    # ------------------------------------------------------
    def select_action(self, board, player, legal_moves, greedy: bool = False):
        normalized_moves, mapping = self._prepare_legal(legal_moves)
        if not normalized_moves:
            return None

        mask = self.action_manager.make_legal_action_mask(normalized_moves)

        state = self.encoder.encode(board, player=player)
        if state.dim() == 3:
            state = state.unsqueeze(0)
        state = state.to(self.device)

        q = self.model.get_q_values(state)
        masked = q.clone()
        masked[mask.unsqueeze(0) == 0] = -1e9

        a = int(torch.argmax(masked, dim=1).item())

        # If greedy flag is set, we already forced argmax; return mapped env action
        if greedy:
            return mapping.get(a)

        # Directly return the env_action we stored
        return mapping.get(a)
