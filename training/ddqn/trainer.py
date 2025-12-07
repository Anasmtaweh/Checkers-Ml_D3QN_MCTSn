import random
from typing import Optional, Dict, Any, List, Tuple, Callable, Union

import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401  (kept if you later want MSELoss etc.)

from training.ddqn.model import DDQNModel
from training.common.board_encoder import CheckersBoardEncoder
from training.common.replay_buffer import ReplayBuffer
from training.common.action_manager import ActionManager

Move = Tuple[Tuple[int, int], Tuple[int, int]]


class DDQNTrainer:
    def __init__(
        self,
        env: Any,
        model: DDQNModel,
        encoder: Optional[CheckersBoardEncoder] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        action_manager: Optional[ActionManager] = None,
        device: Union[str, torch.device] = "cpu",
        gamma: float = 0.99,
        batch_size: int = 64,
        lr: float = 1e-3,
        target_update_interval: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        replay_warmup_size: int = 1_000,
        max_steps_per_episode: int = 300,
        use_soft_update: bool = False,
        soft_update_tau: float = 0.0,
        debug: bool = False,
        lr_schedule: str = "none",
        lr_gamma: float = 0.99,
        q_clip: float = 0.0,
    ):
        self.env = env
        self.model: DDQNModel = model
        self.device = torch.device(device)

        # Components
        self.encoder = encoder or CheckersBoardEncoder()
        self.replay_buffer = replay_buffer or ReplayBuffer(
            capacity=50_000,
            device=self.device,
            action_dim=model.action_dim,
        )
        self.action_manager = action_manager or ActionManager(device=self.device)
        self.action_manager.device = self.device

        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.target_update_interval = target_update_interval
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.replay_warmup_size = replay_warmup_size
        self.max_steps_per_episode = max_steps_per_episode
        self.use_soft_update = use_soft_update
        self.soft_update_tau = soft_update_tau
        self.debug = debug
        self.lr_schedule_type = lr_schedule
        self.lr_gamma = lr_gamma
        self.q_clip = q_clip

        self.optimizer = torch.optim.Adam(self.model.online.parameters(), lr=self.lr)
        self.lr_scheduler = None
        self.criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.nn.SmoothL1Loss()
        self.global_step = 0
        self.epsilon = epsilon_start
        self.train_step_count = 0
        self.metric_writer: Optional[Any] = None  # will be set to DDQNMetricWriter in train_ddqn.py

        # Keep model/device aligned
        self.model.device = self.device
        self._init_scheduler()

    # ------------------------------------------------------------------
    # Setup / helpers
    # ------------------------------------------------------------------

    def _init_scheduler(self) -> None:
        if self.lr_schedule_type == "exponential":
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.lr_gamma,
            )
        elif self.lr_schedule_type == "cosine":
            # Large T_max for slow cosine decay; tune if needed
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=100_000,
            )

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _update_epsilon(self) -> None:
        decay_ratio = min(1.0, max(0.0, self.global_step / float(self.epsilon_decay_steps)))
        self.epsilon = self.epsilon_start * (1.0 - decay_ratio) + self.epsilon_end * decay_ratio
        self.epsilon = max(self.epsilon_end, min(self.epsilon_start, self.epsilon))

    def _encode_state(self, board: np.ndarray, player: int, info: Optional[Dict[str, Any]]) -> torch.Tensor:
        encoded = self.encoder.encode(board, player=player, info=info)
        if encoded.dim() == 3:
            encoded = encoded.unsqueeze(0)
        return encoded.to(self.device)

    # ------------------------------------------------------------------
    # Single DDQN update
    # ------------------------------------------------------------------

    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.replay_warmup_size:
            return None
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].float().to(self.device)

        legal_masks_next = batch.get("legal_masks_next")
        if legal_masks_next is not None:
            legal_masks_next = legal_masks_next.to(self.device)
            if self.debug:
                assert legal_masks_next.shape[1] == self.action_manager.action_dim

        legal_masks_current = batch.get("legal_masks_current")
        if legal_masks_current is not None:
            legal_masks_current = legal_masks_current.to(self.device)
            if self.debug:
                assert legal_masks_current.shape[1] == self.action_manager.action_dim

        # Q(s,·)
        q_values = self.model.online(states)
        if self.q_clip > 0:
            q_values = q_values.clamp(-self.q_clip, self.q_clip)

        if legal_masks_current is not None:
            q_values = q_values.clone()
            q_values[legal_masks_current == 0] = -1e9

        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        if self.debug:
            assert not torch.isnan(q_values).any()

        with torch.no_grad():
            # a* = argmax_a Q_online(s', a)
            q_next_online = self.model.online(next_states)
            if self.q_clip > 0:
                q_next_online = q_next_online.clamp(-self.q_clip, self.q_clip)
            if legal_masks_next is not None:
                q_next_online = q_next_online.clone()
                q_next_online[legal_masks_next == 0] = -1e9
            a_next = q_next_online.argmax(dim=1)

            # Q_target(s', a*)
            q_next_target = self.model.target(next_states)
            if self.q_clip > 0:
                q_next_target = q_next_target.clamp(-self.q_clip, self.q_clip)
            q_target_selected = q_next_target.gather(1, a_next.unsqueeze(1)).squeeze(1)

            td_target = rewards + (1.0 - dones) * self.gamma * q_target_selected

        loss = self.criterion(q_selected, td_target.detach())

        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: loss is NaN or Inf; skipping update")
            return None

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.online.parameters(), 1.0)
        self.optimizer.step()

        self.train_step_count += 1
        if self.metric_writer is not None:
            self.metric_writer.log_loss(self.train_step_count, loss.item())

        if self.use_soft_update and self.soft_update_tau > 0:
            self.model.soft_update_target(self.soft_update_tau)
        elif self.train_step_count % self.target_update_interval == 0:
            self.model.update_target()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Legal moves processing (shared by normal / forced continuation)
    # Step-by-step multi-captures: we only execute ONE capture step at a time.
    # ------------------------------------------------------------------

    def _build_legal_from_raw(
        self,
        legal_moves_raw: List[Any],
    ) -> Tuple[Dict[int, Any], List[Move], bool]:
        """
        Parse env-format moves into:
          - legal_index_to_move: action_idx -> env action (simple move or single capture step)
          - normalized_moves: List[(from, to)] used for action masking / indexing
          - has_capture: whether any capture exists

        Design:
        - Simple quiet move: ((r1,c1), (r2,c2))  -> env action is same pair
        - Single capture step: ((r1,c1), (r2,c2), (jr,jc)) -> env action is that triple
        - Multi-step sequence: [step0, step1, ...] (each step is a triple)
            -> we treat ONLY step0 as the RL decision for this turn
        - If any capture exists, ALL non-captures are dropped (mandatory capture).
        """
        legal_index_to_move: Dict[int, Any] = {}
        normalized_moves: List[Move] = []
        has_capture = False

        # (env_action, norm, is_capture)
        candidates: List[Tuple[Any, Move, bool]] = []

        for mv in legal_moves_raw:
            if not isinstance(mv, (list, tuple)):
                continue

            env_action: Optional[Any] = None
            norm: Optional[Move] = None
            is_capture = False

            # Case 1: multi-step capture sequence: [ (from, to, jumped), ... ]
            if (
                isinstance(mv, (list, tuple))
                and mv
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

            # Case 3: simple move: ((r1,c1), (r2,c2))
            elif (
                len(mv) == 2
                and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in mv)
            ):
                (r1, c1), (r2, c2) = mv
                r1, c1 = int(r1), int(c1)
                r2, c2 = int(r2), int(c2)
                norm = ((r1, c1), (r2, c2))
                env_action = ((r1, c1), (r2, c2))
                # simple moves are not captures here; capture moves should be emitted as triples by rules

            if norm is None or env_action is None:
                continue

            candidates.append((env_action, norm, is_capture))
            if is_capture:
                has_capture = True

        # Second pass: enforce mandatory capture & build indexed mapping
        for env_action, norm, is_capture in candidates:
            if has_capture and not is_capture:
                # non-capturing moves are illegal when a capture exists
                continue
            try:
                idx = self.action_manager.encode_move(norm)
            except ValueError:
                continue
            legal_index_to_move[idx] = env_action
            normalized_moves.append(norm)

        return legal_index_to_move, normalized_moves, has_capture

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------

    def run_episode(self, training: bool = True, max_moves: Optional[int] = None) -> Dict[str, Any]:
        max_moves = max_moves or self.max_steps_per_episode

        board = self.env.reset()
        player = getattr(self.env, "current_player", 1)
        base_player = player  # we always store rewards from POV of the initial side
        info: Dict[str, Any] = {}
        done = False
        moves = 0
        total_reward = 0.0
        last_loss: Optional[float] = None

        while not done and moves < max_moves:
            # New turn: clear any multi-jump bookkeeping (if you ever use it elsewhere)
            self.action_manager.multi_jump_map.clear()
            self.action_manager.reverse_multi_jump_map.clear()

            legal_moves_raw = (
                self.env.get_legal_moves()
                if hasattr(self.env, "get_legal_moves")
                else self.env.legal_moves()
            )
            if not legal_moves_raw:
                break

            legal_index_to_move, normalized_moves, has_capture = self._build_legal_from_raw(legal_moves_raw)
            if self.debug and moves < 2:
                print(
                    f"[DEBUG] step={self.global_step} legal_raw={len(legal_moves_raw)} "
                    f"normalized={len(normalized_moves)} has_capture={has_capture}"
                )

            if not normalized_moves:
                break

            mask = self.action_manager.make_legal_action_mask(normalized_moves).to(self.device)
            legal_indices = list(legal_index_to_move.keys())

            state_tensor = self.encoder.encode(board, player=player, info=info)
            if state_tensor.dim() == 3:
                state_tensor = state_tensor.unsqueeze(0)
            state_tensor = state_tensor.to(self.device)

            epsilon = self.epsilon if training else 0.0

            # Forced continuation of capture chain: no exploration, recompute legal set
            if training and info.get("continue", False):
                legal_moves_raw = (
                    self.env.get_legal_moves()
                    if hasattr(self.env, "get_legal_moves")
                    else self.env.legal_moves()
                )
                self.action_manager.multi_jump_map.clear()
                self.action_manager.reverse_multi_jump_map.clear()
                legal_index_to_move, normalized_moves, has_capture = self._build_legal_from_raw(legal_moves_raw)
                if not normalized_moves:
                    break
                mask = self.action_manager.make_legal_action_mask(normalized_moves).to(self.device)
                q_values = self.model.get_q_values(state_tensor)
                masked_q = q_values.clone()
                masked_q[mask.unsqueeze(0) == 0] = -1e9
                action_idx = int(torch.argmax(masked_q, dim=1).item())
            elif training and random.random() < epsilon:
                action_idx = int(random.choice(legal_indices))
            else:
                q_values = self.model.get_q_values(state_tensor)
                masked_q = q_values.clone()
                masked_q[mask.unsqueeze(0) == 0] = -1e9
                action_idx = int(torch.argmax(masked_q, dim=1).item())
                if self.debug and (self.global_step % 500 == 0):
                    print(
                        f"[DEBUG] step={self.global_step} q_shape={q_values.shape} "
                        f"mask_shape={mask.shape} legal={len(legal_indices)} selected={action_idx}"
                    )
                    assert not torch.isnan(masked_q).any()

            action_idx_int = int(action_idx)
            env_action = legal_index_to_move.get(
                action_idx_int,
                self.action_manager.index_to_move(action_idx_int),
            )

            # Step env with our chosen move (pair or single capture step triple)
            step_result = self.env.step(env_action)
            if len(step_result) == 5:
                next_board, next_player, reward, done, info = step_result
            else:
                next_board, reward, done, info = step_result
                next_player = getattr(self.env, "current_player", player)

            # Next state
            next_state_tensor = self.encoder.encode(next_board, player=next_player, info=info)
            if next_state_tensor.dim() == 3:
                next_state_tensor = next_state_tensor.unsqueeze(0)
            next_state_tensor = next_state_tensor.to(self.device)

            # ------------------------------------------------------------------
            # Reward: use ENV rewards only (your choice = Option 1)
            # ------------------------------------------------------------------
            shaped_reward = float(reward)

            # Align reward to base_player POV so stats are from a fixed side
            reward_pov = shaped_reward if player == base_player else -shaped_reward

            # ------------------------------------------------------------------
            # Store transition + train
            # ------------------------------------------------------------------
            if training:
                legal_mask_next = None
                if not done:
                    if hasattr(self.env, "get_legal_moves"):
                        legal_moves_next_raw = self.env.get_legal_moves()
                    else:
                        legal_moves_next_raw = self.env.legal_moves()

                    # For next-state mask we only need normalized (from,to) moves, no env-action format
                    _, next_normalized_moves, _ = self._build_legal_from_raw(legal_moves_next_raw)
                    if next_normalized_moves:
                        legal_mask_next = self.action_manager.make_legal_action_mask(next_normalized_moves)

                self.replay_buffer.add(
                    state_tensor,
                    action_idx,
                    reward_pov,
                    next_state_tensor,
                    done,
                    legal_mask_next=legal_mask_next,
                    legal_mask_current=mask,
                )
                loss = self.train_step()
                last_loss = loss if loss is not None else last_loss

            total_reward += reward_pov
            moves += 1
            self.global_step += 1
            if training:
                self._update_epsilon()
            player = next_player
            board = next_board

        return {
            "moves": moves,
            "total_reward": total_reward,
            "winner": info.get("winner", None),
            "training": training,
            "loss": last_loss,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        num_episodes: int,
        max_moves: Optional[int] = None,
        log_interval: int = 10,
        eval_interval: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        for ep in range(1, num_episodes + 1):
            stats = self.run_episode(training=True, max_moves=max_moves)
            history.append(stats)

            if self.metric_writer is not None:
                self.metric_writer.log_episode(
                    episode=ep,
                    reward=stats["total_reward"],
                    epsilon=self.epsilon,
                    replay_size=len(self.replay_buffer),
                    moves=stats["moves"],
                    loss=stats.get("loss"),
                )

            if ep % log_interval == 0:
                print(
                    f"[Episode {ep}] moves={stats['moves']} reward={stats['total_reward']:.2f} "
                    f"epsilon={self.epsilon:.3f} replay_size={len(self.replay_buffer)}"
                )

            if eval_interval and ep % eval_interval == 0:
                if hasattr(self, "eval_callback") and callable(getattr(self, "eval_callback")):
                    try:
                        getattr(self, "eval_callback")(self, ep)
                    except Exception:
                        pass

        return history

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "model": {
                    "online": self.model.online.state_dict(),
                    "target": self.model.target.state_dict(),
                },
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "global_step": self.global_step,
            },
            path,
        )

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        model_state = state.get("model", {})
        if model_state:
            self.model.online.load_state_dict(model_state.get("online", {}))
            self.model.target.load_state_dict(model_state.get("target", {}))
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.epsilon = state.get("epsilon", self.epsilon)
        self.global_step = state.get("global_step", self.global_step)
