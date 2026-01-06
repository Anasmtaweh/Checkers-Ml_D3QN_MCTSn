import os
import pickle
import time
from collections import deque
from typing import Any, Dict, List, Optional
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ray

# Force Ray configuration
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["RAY_DISABLE_METRICS_COLLECTION"] = "1"

# ============================================================================
# RAY ACTOR WORKER (CPU MODE - THE SPEED KING)
# ============================================================================

# We set num_gpus=0 to force Ray to put this on CPU cores.
@ray.remote(num_gpus=0)
class SelfPlayWorker:
    def __init__(self, action_dim: int):
        import torch
        from mcts_workspace.core.action_manager import ActionManager
        from mcts_workspace.core.board_encoder import CheckersBoardEncoder
        from mcts_workspace.training.alpha_zero.mcts import MCTS
        from mcts_workspace.training.alpha_zero.network import AlphaZeroModel

        # FORCE CPU
        self.device = "cpu"

        self.action_manager = ActionManager(self.device)
        self.encoder = CheckersBoardEncoder()
        
        self.model = AlphaZeroModel(action_dim, self.device)
        self.model.eval()

        self.mcts = MCTS(
            model=self.model,
            action_manager=self.action_manager,
            encoder=self.encoder,
            c_puct=3.0,
            num_simulations=100,
            device=self.device
        )

    @staticmethod
    def _compute_dirichlet_alpha(n_legal: int, base_alpha: float, ref_moves: float = 6.0, min_alpha: float = 0.03) -> float:
        if n_legal <= 1: return 0.0
        scaled = base_alpha * (ref_moves / float(n_legal))
        return float(max(min_alpha, min(base_alpha, scaled)))

    def play_game(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from mcts_workspace.core.game import CheckersEnv

        # Load weights to CPU
        self.model.network.load_state_dict(params["model_state_dict"])
        self.model.eval()

        # Update MCTS flags
        self.mcts.c_puct = params.get("c_puct", 3.0)
        self.mcts.num_simulations = params["num_sims"]
        self.mcts.dirichlet_epsilon = params["dirichlet_epsilon"]
        self.mcts.draw_value = params.get("mcts_draw_value", 0.0)
        self.mcts.search_draw_bias = params.get("search_draw_bias", 0.0)
        self.mcts.skip_root_sims_on_forced = True

        env = CheckersEnv(
            max_moves=params["env_max_moves"],
            no_progress_limit=params["no_progress_plies"],
        )
        env.reset()

        states, players, policies = [], [], []
        move_count = 0
        winner = 0
        move_mapping_failures = 0
        base_alpha = float(params.get("dirichlet_alpha", 0.20))

        while not env.done:
            move_count += 1
            if move_count > (params["env_max_moves"] * 1.5):
                env.done = True; env.winner = 0; break

            legal_moves = env.get_legal_moves()
            if not legal_moves:
                _, winner = env._check_game_over()
                break

            is_forced = (len(legal_moves) == 1)
            temp = 1.0 if move_count <= params["temp_threshold"] else 0.0
            is_exploring = temp > 0

            if is_exploring and not is_forced:
                n_legal = len(legal_moves)
                self.mcts.dirichlet_alpha = self._compute_dirichlet_alpha(n_legal, base_alpha)
            else:
                self.mcts.dirichlet_alpha = 0.0

            # CPU MCTS CALL (No "with torch.no_grad" needed, usually faster without overhead)
            action_probs, _ = self.mcts.get_action_prob(env, temp=temp, training=is_exploring)

            if not is_forced:
                board = env.board.get_state()
                player = env.current_player
                encoded_state = self.encoder.encode(board, player, force_move_from=env.force_capture_from)
                states.append(encoded_state)
                players.append(player)
                policies.append(action_probs)

            if temp == 0.0:
                action_id = int(np.argmax(action_probs))
            else:
                action_id = int(np.random.choice(len(action_probs), p=action_probs))

            move = self.mcts._get_move_from_action(action_id, legal_moves, player=env.current_player)

            if move is None:
                move_mapping_failures += 1
                move = legal_moves[0] if legal_moves else None

            if move is None:
                winner = -env.current_player
                break

            _, _, done, info = env.step(move)
            if done: winner = info["winner"]

        return {
            "states": states,
            "players": players,
            "policies": policies,
            "winner": winner,
            "move_mapping_failures": move_mapping_failures,
        }


# ============================================================================
# TRAINER
# ============================================================================

class AlphaZeroTrainer:
    def __init__(
        self,
        model,
        mcts,
        action_manager,
        board_encoder,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
        buffer_size: int = 10000,
        batch_size: int = 256,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        value_loss_weight: float = 0.25,
        policy_loss_weight: float = 1.0,
        temp_threshold: int = 30,
        draw_penalty: float = 0.0,
        env_max_moves: int = 200,
        no_progress_plies: int = 80,
        dirichlet_epsilon: float = 0.25,
        num_ray_workers: int = 4,
    ):
        self.model = model
        self.mcts = mcts
        self.action_manager = action_manager
        self.board_encoder = board_encoder
        self.device = device
        self.batch_size = batch_size
        self.value_loss_weight = value_loss_weight
        self.policy_loss_weight = policy_loss_weight
        self.temp_threshold = temp_threshold
        self.draw_penalty = draw_penalty
        self.env_max_moves = int(env_max_moves)
        self.no_progress_plies = int(no_progress_plies)
        self.dirichlet_epsilon = float(dirichlet_epsilon)
        self.num_ray_workers = int(num_ray_workers)
        self._ray_workers = None
        self.replay_buffer: deque = deque(maxlen=buffer_size)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.network.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optimizer

        self.training_stats = {"total_games": 0, "total_steps": 0, "losses": [], "value_losses": [], "policy_losses": []}

    def _ensure_ray_workers(self):
        if self._ray_workers is not None:
            if len(self._ray_workers) == self.num_ray_workers:
                return
        
        if not ray.is_initialized():
            # Initializing Ray for local PC
            ray.init(ignore_reinit_error=True, log_to_driver=False)
            
        print(f"  [Ray] Initializing {self.num_ray_workers} workers on CPU...")
        action_dim = self.action_manager.action_dim
        self._ray_workers = [SelfPlayWorker.remote(action_dim=action_dim) for _ in range(self.num_ray_workers)]

    def self_play(self, num_games: int, verbose: bool = True, iteration: int = 0) -> Dict[str, Any]:
        self._ensure_ray_workers()
        if self._ray_workers is None:
            raise RuntimeError("Ray workers failed to initialize")
        
        # Prepare params (Convert weights to CPU first to avoid CUDA serialization)
        model_state = self.model.network.state_dict()
        cpu_state = {k: v.cpu() for k, v in model_state.items()}

        params = {
            "model_state_dict": cpu_state,
            "action_dim": self.action_manager.action_dim,
            "c_puct": getattr(self.mcts, "c_puct", 3.0),
            "num_sims": self.mcts.num_simulations,
            "temp_threshold": self.temp_threshold,
            "dirichlet_alpha": getattr(self.mcts, "dirichlet_alpha", 0.3),
            "dirichlet_epsilon": self.dirichlet_epsilon,
            "env_max_moves": self.env_max_moves,
            "no_progress_plies": self.no_progress_plies,
            "mcts_draw_value": getattr(self.mcts, "draw_value", 0.0),
            "search_draw_bias": getattr(self.mcts, "search_draw_bias", 0.0),
            "skip_root_sims_on_forced": True
        }

        futures = []
        for i in range(num_games):
            w = self._ray_workers[i % len(self._ray_workers)]
            futures.append(w.play_game.remote(params))  # type: ignore

        # --- LIVE PROGRESS LOGGING ---
        results = []
        pending = futures
        start_time = time.time()
        
        if verbose:
            sys.stdout.write(f"  Launched {num_games} games. Waiting...\n")

        while pending:
            done_ids, pending = ray.wait(pending, num_returns=1)
            for result_id in done_ids:
                res = ray.get(result_id)
                results.append(res)
                
                if verbose:
                    elapsed = time.time() - start_time
                    rate = len(results) / max(1e-5, elapsed)
                    sys.stdout.write(f"\r  » Progress: {len(results)}/{num_games} | Speed: {rate:.2f} games/s")
                    sys.stdout.flush()
        
        if verbose:
            sys.stdout.write("\n")

        # Stats Processing
        stats: Dict[str, Any] = {"p1_wins": 0, "p2_wins": 0, "draws": 0, "total_moves": 0, "move_mapping_failures": 0}

        for game in results:
            stats["total_moves"] += len(game["states"])
            stats["move_mapping_failures"] += int(game.get("move_mapping_failures", 0))

            winner = game["winner"]
            if winner == 1: stats["p1_wins"] += 1
            elif winner == -1: stats["p2_wins"] += 1
            else: stats["draws"] += 1

            self._process_game_data(game)

        stats["avg_game_length"] = stats["total_moves"] / max(1, num_games)
        stats["buffer_size"] = len(self.replay_buffer)
        stats["p1_win_rate"] = stats["p1_wins"] / max(1, num_games)
        stats["p2_win_rate"] = stats["p2_wins"] / max(1, num_games)
        stats["draw_rate"] = stats["draws"] / max(1, num_games)

        self.training_stats["total_games"] += num_games
        return stats

    def _process_game_data(self, game_data: Dict[str, Any]) -> None:
        states = game_data["states"]
        players = game_data["players"]
        policies = game_data["policies"]
        winner = game_data["winner"]

        for i in range(len(states)):
            player = players[i]
            if winner == 1: z = 1.0 if player == 1 else -1.0
            elif winner == -1: z = 1.0 if player == -1 else -1.0
            else: z = 0.0
            
            self.replay_buffer.append((states[i].cpu(), policies[i], z))

    def train_step(self, epochs: int = 1, verbose: bool = True) -> Dict[str, float]:
        if len(self.replay_buffer) < self.batch_size:
            if verbose: print(f"Insufficient data: {len(self.replay_buffer)}/{self.batch_size}")
            return {"loss": 0.0, "value_loss": 0.0, "policy_loss": 0.0}

        self.model.train()
        losses, v_losses, p_losses = [], [], []

        for _ in range(epochs):
            batch = self._sample_batch(self.batch_size)
            states = batch["states"].to(self.device)
            policy_targets = batch["policy_targets"].to(self.device)
            value_targets = batch["value_targets"].to(self.device)

            policy_logits, value_pred = self.model.get_policy_value(states)

            value_loss = self._compute_value_loss(value_pred, value_targets)
            policy_loss = self._compute_policy_loss(policy_logits, policy_targets)
            loss = (self.value_loss_weight * value_loss) + (self.policy_loss_weight * policy_loss)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            losses.append(loss.item())
            v_losses.append(value_loss.item())
            p_losses.append(policy_loss.item())

        return {
            "loss": float(np.mean(losses)),
            "value_loss": float(np.mean(v_losses)),
            "policy_loss": float(np.mean(p_losses))
        }

    def _sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        states_list, policies_list, values_list = [], [], []

        for idx in indices:
            state, policy, value = self.replay_buffer[idx]
            states_list.append(state)
            policies_list.append(policy)
            values_list.append(value)

        return {
            "states": torch.stack(states_list),
            "policy_targets": torch.tensor(np.array(policies_list), dtype=torch.float32),
            "value_targets": torch.tensor(values_list, dtype=torch.float32).unsqueeze(1)
        }

    def _compute_value_loss(self, value_pred: torch.Tensor, value_target: torch.Tensor) -> torch.Tensor:
        return nn.MSELoss()(value_pred, value_target)

    def _compute_policy_loss(self, policy_logits: torch.Tensor, policy_target: torch.Tensor) -> torch.Tensor:
        return -(policy_target * policy_logits).sum(dim=1).mean()

    def save_checkpoint(self, path: str, iteration: int = 0, additional_info: Optional[Dict] = None):
        checkpoint = {
            "iteration": iteration,
            "model_state_dict": self.model.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_stats": self.training_stats,
            "buffer_size": len(self.replay_buffer),
        }
        if additional_info: checkpoint.update(additional_info)
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> Dict:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_stats = checkpoint.get("training_stats", self.training_stats)
        print(f"✓ Checkpoint loaded from {path}")
        return checkpoint

    def save_replay_buffer(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.replay_buffer, f)
        print(f"✓ Replay buffer saved to {path}")

    def load_replay_buffer(self, path: str):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.replay_buffer = pickle.load(f)
            print(f"✓ Replay buffer loaded ({len(self.replay_buffer)} samples)")
        else:
            print(f"⚠️ Replay buffer not found at {path}")

    def get_buffer_size(self) -> int:
        return len(self.replay_buffer)