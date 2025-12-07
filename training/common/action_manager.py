from typing import Dict, Tuple, List, Any, Union

import torch

Move = Tuple[Tuple[int, int], Tuple[int, int]]


class ActionManager:
    """
    Fixed action mapping for DDQN: all ordered origin/destination pairs on 8x8 board (64*63=4032).
    """

    def __init__(self, device: Union[str, torch.device], action_dim: int = 4032):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.action_dim = action_dim

        self._index_to_pair: List[Tuple[int, int]] = []
        self._pair_to_index: Dict[Tuple[int, int], int] = {}
        # Multi-jump sequence mapping: action_idx -> full sequence, and reverse
        self.multi_jump_map: Dict[int, Tuple[Tuple[int, int], ...]] = {}
        self.reverse_multi_jump_map: Dict[Tuple[Tuple[int, int], ...], int] = {}

        for origin in range(64):
            for dest in range(64):
                if origin == dest:
                    continue
                idx = len(self._index_to_pair)
                pair = (origin, dest)
                self._index_to_pair.append(pair)
                self._pair_to_index[pair] = idx

    @staticmethod
    def _coord_to_index(row: int, col: int) -> int:
        return row * 8 + col

    @staticmethod
    def _index_to_coord(idx: int) -> Tuple[int, int]:
        return divmod(idx, 8)

    def encode_move(self, move: Move) -> int:
        (r1, c1), (r2, c2) = move
        origin = self._coord_to_index(r1, c1)
        dest = self._coord_to_index(r2, c2)
        if origin == dest:
            raise ValueError("Origin and destination cannot be the same for a move.")
        try:
            return self._pair_to_index[(origin, dest)]
        except KeyError as e:
            raise ValueError(f"Move out of range: {move}") from e

    def decode_index(self, action_index: int) -> Move:
        if action_index < 0 or action_index >= self.action_dim:
            raise ValueError(f"Action index {action_index} out of bounds [0, {self.action_dim}).")
        origin, dest = self._index_to_pair[action_index]
        r1, c1 = self._index_to_coord(origin)
        r2, c2 = self._index_to_coord(dest)
        return (r1, c1), (r2, c2)

    # Alias for backward compatibility with trainer usage
    def index_to_move(self, action_index: int) -> Move:
        return self.decode_index(action_index)

    def make_legal_action_mask(self, legal_moves: List[Move]) -> torch.Tensor:
        mask = torch.zeros(self.action_dim, dtype=torch.float32, device=self.device)
        for mv in legal_moves:
            try:
                start = (int(mv[0][0]), int(mv[0][1]))
                end = (int(mv[1][0]), int(mv[1][1]))
                idx = self.encode_move((start, end))
                mask[idx] = 1.0
            except ValueError:
                continue
        return mask

    def legal_moves_to_indices(self, legal_moves: List[Move]) -> List[int]:
        """
        Convert a list of legal move tuples into a list of flat action indices.
        """
        indices: List[int] = []
        for mv in legal_moves:
            try:
                start = (int(mv[0][0]), int(mv[0][1]))
                end = (int(mv[1][0]), int(mv[1][1]))
                idx = self.encode_move((start, end))
                indices.append(idx)
            except ValueError:
                continue
        return indices

    def encode_multi_jump(self, sequence: List[Tuple[int, int]]) -> int:
        """
        Encode a full multi-jump sequence into an action index (uses first hop for indexing).
        """
        if len(sequence) < 2:
            raise ValueError("Multi-jump sequence must contain at least two coordinates.")
        start = (int(sequence[0][0]), int(sequence[0][1]))
        nxt = (int(sequence[1][0]), int(sequence[1][1]))
        norm: Move = (start, nxt)
        action_idx = self.encode_move(norm)
        seq_tuple: Tuple[Tuple[int, int], ...] = tuple((int(p[0]), int(p[1])) for p in sequence)
        # Store the full sequence as a key to avoid collisions when multiple paths share the first hop
        self.multi_jump_map[action_idx] = seq_tuple
        self.reverse_multi_jump_map[seq_tuple] = action_idx
        return action_idx

    def decode_multi_jump(self, action_idx: int) -> Any:
        """
        Decode an action index back to a stored multi-jump sequence if present; otherwise simple move.
        """
        if action_idx in self.multi_jump_map:
            return self.multi_jump_map[action_idx]
        return self.index_to_move(action_idx)
