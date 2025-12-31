import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import List, Dict, Any, Optional
import time
import os

# Fix Ray warnings and metrics errors (must be set before importing ray)
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["RAY_DISABLE_METRICS_COLLECTION"] = "1"

import ray

class AlphaZeroTrainer:
    """
    AlphaZero Trainer: Self-Play + Neural Network Training.

    This class orchestrates the entire AlphaZero training pipeline:
    1. Self-Play: Generate training data by playing games against itself
    2. Training: Update neural network using generated data
    3. Iteration: Repeat until convergence

    Key Differences from D3QN:
    - Stores full game histories, not single transitions
    - Value targets computed from game outcomes (not bootstrapped)
    - Policy targets from MCTS visit counts (not argmax)
    - Combined policy + value loss function
    """

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
        value_loss_weight: float = 1.0,
        policy_loss_weight: float = 1.0,
        temp_threshold: int = 30,
        draw_penalty: float = -0.1,
        env_max_moves: int = 200,
        no_progress_plies: int = 80,
        dirichlet_epsilon: float = 0.1,
    ):
        """
        Initialize AlphaZero trainer.

        Args:
            model: AlphaZeroModel instance
            mcts: MCTS instance for self-play
            action_manager: ActionManager for move encoding
            board_encoder: CheckersBoardEncoder for state encoding
            optimizer: Optional custom optimizer (default: Adam)
            device: Device for training
            buffer_size: Maximum replay buffer size
            batch_size: Training batch size
            lr: Learning rate
            weight_decay: L2 regularization
            value_loss_weight: Weight for value loss in total loss
            policy_loss_weight: Weight for policy loss in total loss
            temp_threshold: Move number to switch from temp=1.0 to temp=0.0
            draw_penalty: Value target for drawn games (default: -0.1)
        """
        self.model = model
        self.mcts = mcts
        self.action_manager = action_manager
        self.board_encoder = board_encoder
        self.device = device

        # Training hyperparameters
        self.batch_size = batch_size
        self.value_loss_weight = value_loss_weight
        self.policy_loss_weight = policy_loss_weight
        self.temp_threshold = temp_threshold
        self.draw_penalty = draw_penalty
        self.env_max_moves = int(env_max_moves)
        self.no_progress_plies = int(no_progress_plies)

        # Replay buffer: stores (state, policy_target, value_target) tuples
        self.replay_buffer: deque = deque(maxlen=buffer_size)

        # Optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.network.parameters(),
                lr=lr,
                weight_decay=1e-3,
            )
        else:
            self.optimizer = optimizer

        # Statistics tracking
        self.training_stats = {
            "total_games": 0,
            "total_steps": 0,
            "losses": [],
            "value_losses": [],
            "policy_losses": [],
        }

    # ================================================================
    # SELF-PLAY: Generate Training Data
    # ================================================================

    def self_play(self, num_games: int, verbose: bool = True) -> Dict[str, Any]:
        """Self-play using Ray for parallel GPU execution."""
        start_time = time.time()

        # Initialize Ray (Reduced resources for stability)
        if not ray.is_initialized():
            ray.init(
                num_cpus=10,  # Increase to 10 to feed 4 hungry workers
                num_gpus=1,  # Must be INTEGER (physical hardware)
                include_dashboard=False,
                ignore_reinit_error=True,
                logging_level="ERROR",
                log_to_driver=False,
                _system_config={
                    "metrics_report_interval_ms": 0,
                },
            )

        if verbose:
            print("  Using Ray for parallel self-play (4 workers, GPU-accelerated)")

        model_state = self.model.network.state_dict()

        # 0.22 GPU allows 4 workers (0.88 total usage)
        @ray.remote(num_gpus=0.22)
        def play_game_remote(params: Dict[str, Any]) -> Dict[str, Any]:
            """Remote worker for parallel self-play games.
            
            Args:
                params: Dictionary containing:
                    - model_state_dict: Model weights
                    - action_dim: Action space dimension
                    - c_puct: MCTS exploration constant
                    - num_sims: Number of MCTS simulations
                    - temp_threshold: Move number for temperature switch
                    - dirichlet_alpha: Dirichlet noise alpha
                    - dirichlet_epsilon: Dirichlet noise epsilon
                    - env_max_moves: Maximum moves per game
                    - no_progress_plies: No-progress limit
                    - mcts_draw_value: Draw value for MCTS
                    - search_draw_bias: Search-time draw bias
            """
            import torch
            import numpy as np
            from core.game import CheckersEnv
            from core.action_manager import ActionManager
            from core.board_encoder import CheckersBoardEncoder
            from training.alpha_zero.network import AlphaZeroModel
            from training.alpha_zero.mcts import MCTS

            device = "cuda" if torch.cuda.is_available() else "cpu"

            action_manager = ActionManager(device)
            encoder = CheckersBoardEncoder()
            model = AlphaZeroModel(params["action_dim"], device)
            model.network.load_state_dict(params["model_state_dict"])
            model.eval()

            mcts = MCTS(
                model=model,
                action_manager=action_manager,
                encoder=encoder,
                c_puct=params["c_puct"],
                num_simulations=params["num_sims"],
                device=device,
                dirichlet_alpha=params["dirichlet_alpha"],
                dirichlet_epsilon=params["dirichlet_epsilon"],
                draw_value=params["mcts_draw_value"],
                search_draw_bias=params["search_draw_bias"],
            )

            env = CheckersEnv(max_moves=params["env_max_moves"], no_progress_limit=params["no_progress_plies"])
            env.reset()

            states, players, policies = [], [], []
            move_count = 0
            winner = 0
            move_mapping_failures = 0

            while not env.done:
                move_count += 1

                # Adaptive exploration
                if move_count <= params["temp_threshold"]:
                    current_alpha = 0.8
                else:
                    current_alpha = 0.4
                mcts.dirichlet_alpha = current_alpha

                temp = 1.0 if move_count <= params["temp_threshold"] else 0.0
                is_exploring = temp > 0

                action_probs, _ = mcts.get_action_prob(env, temp=temp, training=is_exploring)

                board = env.board.get_state()
                player = env.current_player
                encoded_state = encoder.encode(board, player)

                states.append(encoded_state)
                players.append(player)
                policies.append(action_probs)

                legal_moves = env.get_legal_moves()
                if not legal_moves:
                    # No legal moves -> terminal by rules; ask env for winner
                    _, winner = env._check_game_over()
                    break

                if temp == 0:
                    action_id = int(np.argmax(action_probs))
                else:
                    action_id = int(np.random.choice(len(action_probs), p=action_probs))

                move = mcts._get_move_from_action(action_id, legal_moves, player=env.current_player)  # type: ignore
                if move is None:
                    move_mapping_failures += 1
                    move = legal_moves[0] if legal_moves else None
                    if move is None:
                        # Treat as terminal-loss signal for safety (optional)
                        winner = -env.current_player
                        break

                _, _, done, info = env.step(move)
                if done:
                    winner = info["winner"]
                    break

            return {
                "states": states,
                "players": players,
                "policies": policies,
                "winner": winner,
                "move_mapping_failures": move_mapping_failures,
            }

        futures = []
        for _ in range(num_games):
            params = {
                "model_state_dict": model_state,
                "action_dim": self.action_manager.action_dim,
                "c_puct": self.mcts.c_puct,
                "num_sims": self.mcts.num_simulations,
                "temp_threshold": self.temp_threshold,
                "dirichlet_alpha": self.mcts.dirichlet_alpha,
                "dirichlet_epsilon": self.mcts.dirichlet_epsilon,
                "env_max_moves": self.env_max_moves,
                "no_progress_plies": self.no_progress_plies,
                "mcts_draw_value": getattr(self.mcts, "draw_value", self.draw_penalty),
                "search_draw_bias": getattr(self.mcts, "search_draw_bias", -0.03),
            }
            futures.append(play_game_remote.remote(params))  # type: ignore

        results = ray.get(futures)

        stats: Dict[str, Any] = {
            "games_played": 0,
            "p1_wins": 0,
            "p2_wins": 0,
            "draws": 0,
            "total_moves": 0,
            "move_mapping_failures": 0,
        }

        for game_data in results:
            stats["games_played"] += 1
            stats["total_moves"] += len(game_data["states"])
            stats["move_mapping_failures"] += int(game_data.get("move_mapping_failures", 0))

            winner = game_data["winner"]
            if winner == 1:
                stats["p1_wins"] += 1
            elif winner == -1:
                stats["p2_wins"] += 1
            else:
                stats["draws"] += 1

            self._process_game_data(game_data)

        stats["avg_game_length"] = stats["total_moves"] / stats["games_played"]
        stats["buffer_size"] = len(self.replay_buffer)
        stats["p1_win_rate"] = stats["p1_wins"] / stats["games_played"]
        stats["p2_win_rate"] = stats["p2_wins"] / stats["games_played"]
        stats["draw_rate"] = stats["draws"] / stats["games_played"]

        self.training_stats["total_games"] += num_games

        elapsed = time.time() - start_time

        if verbose:
            print("\n  Self-Play Summary:")
            print(f"    P1 Wins: {stats['p1_wins']} ({stats['p1_win_rate']:.1%})")
            print(f"    P2 Wins: {stats['p2_wins']} ({stats['p2_win_rate']:.1%})")
            print(f"    Draws: {stats['draws']} ({stats['draw_rate']:.1%})")
            print(f"    Avg Game Length: {stats['avg_game_length']:.1f} moves")
            print(f"    Move mapping failures: {stats['move_mapping_failures']}")
            print(f"    Buffer Size: {stats['buffer_size']}")
            print(f"  Self-play completed in {elapsed:.1f}s")

        return stats

    def _play_single_game(self) -> Dict[str, Any]:
        """Play a single self-play game (non-Ray) and collect data."""
        from core.game import CheckersEnv

        env = CheckersEnv(max_moves=self.env_max_moves, no_progress_limit=self.no_progress_plies)
        env.reset()

        states = []
        players = []
        policies = []
        move_count = 0

        while not env.done:
            temp = 1.0 if move_count <= self.temp_threshold else 0.0
            is_exploring = temp > 0

            action_probs, root = self.mcts.get_action_prob(env, temp=temp, training=is_exploring)

            board = env.board.get_state()
            player = env.current_player
            encoded_state = self.board_encoder.encode(board, player)

            states.append(encoded_state)
            players.append(player)
            policies.append(action_probs)

            legal_moves = env.get_legal_moves()

            if temp == 0:
                action_id = int(np.argmax(action_probs))
            else:
                action_id = int(np.random.choice(len(action_probs), p=action_probs))

            move = self._get_move_from_action_id(action_id, legal_moves, player=player)
            if move is None:
                move = legal_moves[0] if legal_moves else None
            if move is None:
                break

            _, _, done, info = env.step(move)
            move_count += 1

        _, winner = env._check_game_over()

        return {
            "states": states,
            "players": players,
            "policies": policies,
            "winner": winner,
        }

    def _process_game_data(self, game_data: Dict) -> None:
        """Process game data and add to replay buffer."""
        states = game_data["states"]
        players = game_data["players"]
        policies = game_data["policies"]
        winner = game_data["winner"]

        for i in range(len(states)):
            player = players[i]

            if winner == 1:
                z = 1.0 if player == 1 else -1.0
            elif winner == -1:
                z = 1.0 if player == -1 else -1.0
            else:
                z = 0.0

            self.replay_buffer.append((states[i].cpu(), policies[i], z))

    def save_replay_buffer(self, path: str):
        """Save the replay buffer to disk."""
        import pickle

        try:
            with open(path, "wb") as f:
                pickle.dump(self.replay_buffer, f)
            print(f"  ✓ Replay buffer saved to {path}")
        except Exception as e:
            print(f"  ❌ Failed to save buffer: {e}")

    def load_replay_buffer(self, path: str):
        """Restore replay buffer from disk."""
        import pickle

        if not os.path.exists(path):
            print(f"  ⚠️ No buffer file found at {path}. Starting fresh.")
            return

        try:
            with open(path, "rb") as f:
                loaded_deque = pickle.load(f)

            self.replay_buffer.clear()
            self.replay_buffer.extend(loaded_deque)

            print(f"  ✓ Restored {len(self.replay_buffer)} memories from disk")
        except Exception as e:
            print(f"  ❌ Failed to load buffer: {e}")

    def _get_move_from_action_id(self, action_id: int, legal_moves: List, player: int) -> Optional[Any]:
        """Convert action_id to actual env move (handles canonical flip for player == -1)."""
        move_pair = self.action_manager.get_move_from_id(action_id)

        if move_pair == ((-1, -1), (-1, -1)):
            return None

        # action_id is in canonical (P1) coordinates; flip back for real board when player is -1
        if player == -1:
            move_pair = self.action_manager.flip_move(move_pair)

        for move in legal_moves:
            move_start_landing = self.action_manager._extract_start_landing(move)
            if move_start_landing == move_pair:
                return move

        return None

    # ================================================================
    # TRAINING: Update Neural Network
    # ================================================================

    def train_step(self, epochs: int = 1, verbose: bool = True) -> Dict[str, float]:
        """
        Perform training on replay buffer data.

        AlphaZero Loss:
          L = value_loss_weight * MSE(v_pred, v_target)
            + policy_loss_weight * CE(pi_target, log_pi_pred)
        """
        if len(self.replay_buffer) < self.batch_size:
            if verbose:
                print(f"  Insufficient data: {len(self.replay_buffer)}/{self.batch_size}")
            return {"loss": 0.0, "value_loss": 0.0, "policy_loss": 0.0}

        self.model.train()

        total_loss = 0.0
        total_value_loss = 0.0
        total_policy_loss = 0.0
        num_batches = 0

        for _ in range(epochs):
            batch_data = self._sample_batch(self.batch_size)

            states = batch_data["states"].to(self.device)
            policy_targets = batch_data["policy_targets"].to(self.device)
            value_targets = batch_data["value_targets"].to(self.device)

            policy_logits, value_pred = self.model.get_policy_value(states)

            with torch.no_grad():
                tgt_sum = policy_targets.sum(dim=1).mean().item()
                pred_sum = torch.exp(policy_logits).sum(dim=1).mean().item()
                # Compute policy entropy for diagnostics
                probs = torch.exp(policy_logits)
                entropy = -(probs * policy_logits).sum(dim=1).mean().item()
            print(f"  policy_targets_sum={tgt_sum:.6f}, exp(policy_logits)_sum={pred_sum:.6f}, avg_policy_entropy={entropy:.4f}")

            value_loss = self._compute_value_loss(value_pred, value_targets)
            policy_loss = self._compute_policy_loss(policy_logits, policy_targets)

            loss = (self.value_loss_weight * value_loss) + (self.policy_loss_weight * policy_loss)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_policy_loss = total_policy_loss / num_batches

        self.training_stats["total_steps"] += num_batches
        self.training_stats["losses"].append(avg_loss)
        self.training_stats["value_losses"].append(avg_value_loss)
        self.training_stats["policy_losses"].append(avg_policy_loss)

        if verbose:
            print(
                f"  Training: loss={avg_loss:.4f}, "
                f"value_loss={avg_value_loss:.4f}, "
                f"policy_loss={avg_policy_loss:.4f}"
            )

        return {"loss": avg_loss, "value_loss": avg_value_loss, "policy_loss": avg_policy_loss}

    def _sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch from replay buffer (no replacement)."""
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)

        states_list = []
        policies_list = []
        values_list = []

        for idx in indices:
            state, policy, value = self.replay_buffer[idx]
            states_list.append(state)
            policies_list.append(policy)
            values_list.append(value)

        states = torch.stack(states_list)
        policy_targets = torch.tensor(np.array(policies_list), dtype=torch.float32)
        value_targets = torch.tensor(values_list, dtype=torch.float32).unsqueeze(1)

        return {"states": states, "policy_targets": policy_targets, "value_targets": value_targets}

    def _compute_value_loss(self, value_pred: torch.Tensor, value_target: torch.Tensor) -> torch.Tensor:
        return nn.MSELoss()(value_pred, value_target)

    def _compute_policy_loss(self, policy_logits: torch.Tensor, policy_target: torch.Tensor) -> torch.Tensor:
        # policy_logits are log-probabilities (log_softmax), policy_target is a distribution
        return -(policy_target * policy_logits).sum(dim=1).mean()

    # ================================================================
    # CHECKPOINT MANAGEMENT
    # ================================================================

    def save_checkpoint(self, path: str, iteration: int = 0, additional_info: Optional[Dict] = None):
        checkpoint = {
            "iteration": iteration,
            "model_state_dict": self.model.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_stats": self.training_stats,
            "buffer_size": len(self.replay_buffer),
        }

        if additional_info is not None:
            checkpoint.update(additional_info)

        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> Dict:
        checkpoint = torch.load(path, map_location=self.device)

        self.model.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_stats = checkpoint.get("training_stats", self.training_stats)

        print(f"✓ Checkpoint loaded from {path}")
        print(f"  Iteration: {checkpoint.get('iteration', 'unknown')}")
        print(f"  Total games: {self.training_stats['total_games']}")

        return checkpoint

    # ================================================================
    # UTILITIES
    # ================================================================

    def get_buffer_size(self) -> int:
        return len(self.replay_buffer)

    def clear_buffer(self):
        self.replay_buffer.clear()
        print("✓ Replay buffer cleared")

    def get_training_stats(self) -> Dict:
        return self.training_stats.copy()


# ================================================================
# Training Loop Utilities
# ================================================================

def run_training_iteration(
    trainer: AlphaZeroTrainer,
    iteration: int,
    num_self_play_games: int = 100,
    num_train_epochs: int = 10,
    checkpoint_dir: str = "checkpoints/alphazero",
    save_every: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a single training iteration: Self-Play → Training → Checkpoint."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}")
        print(f"{'='*70}")

    start_time = time.time()

    if verbose:
        print(f"\n[1/2] Self-Play ({num_self_play_games} games)...")
    self_play_stats = trainer.self_play(num_self_play_games, verbose=verbose)

    if verbose:
        print(f"\n[2/2] Training ({num_train_epochs} epochs)...")
    train_stats = trainer.train_step(epochs=num_train_epochs, verbose=verbose)

    if iteration % save_every == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{iteration}.pth")
        trainer.save_checkpoint(
            checkpoint_path,
            iteration=iteration,
            additional_info={"self_play_stats": self_play_stats, "train_stats": train_stats},
        )

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n✓ Iteration {iteration} complete in {elapsed:.1f}s")
        print(f"{'='*70}\n")

    return {
        "iteration": iteration,
        "self_play_stats": self_play_stats,
        "train_stats": train_stats,
        "elapsed_time": elapsed,
    }


# ============================================================================
# MULTIPROCESSING WORKER FUNCTION (Must be at module level)
# ============================================================================

def play_game_worker_func(args):
    """Worker function that plays a single game."""
    import torch
    import numpy as np
    from core.game import CheckersEnv
    from core.action_manager import ActionManager
    from core.board_encoder import CheckersBoardEncoder
    from training.alpha_zero.network import AlphaZeroModel
    from training.alpha_zero.mcts import MCTS

    model_state = args["model_state"]
    action_dim = args["action_dim"]
    c_puct = args["c_puct"]
    num_simulations = args["num_simulations"]
    temp_threshold = args["temp_threshold"]
    dirichlet_alpha = args["dirichlet_alpha"]
    dirichlet_epsilon = args["dirichlet_epsilon"]
    device = args["device"]
    env_max_moves = args.get("env_max_moves", 200)
    no_progress_plies = args.get("no_progress_plies", 80)
    mcts_draw_value = args.get("mcts_draw_value", -0.1)

    action_manager = ActionManager(device)
    encoder = CheckersBoardEncoder()

    model = AlphaZeroModel(action_dim, device)
    model.network.load_state_dict(model_state)
    model.eval()

    mcts = MCTS(
        model=model,
        action_manager=action_manager,
        encoder=encoder,
        c_puct=c_puct,
        num_simulations=num_simulations,
        device=device,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        draw_value=mcts_draw_value,
    )

    env = CheckersEnv(max_moves=env_max_moves, no_progress_limit=no_progress_plies)
    env.reset()

    states = []
    players = []
    policies = []
    move_count = 0
    winner = 0

    while not env.done:
        move_count += 1

        temp = 1.0 if move_count <= temp_threshold else 0.0
        is_exploring = temp > 0

        action_probs, root = mcts.get_action_prob(env, temp=temp, training=is_exploring)

        board = env.board.get_state()
        player = env.current_player
        encoded_state = encoder.encode(board, player)

        states.append(encoded_state)
        players.append(player)
        policies.append(action_probs)

        legal_moves = env.get_legal_moves()
        if temp == 0:
            action_id = int(np.argmax(action_probs))
        else:
            action_id = int(np.random.choice(len(action_probs), p=action_probs))

        # IMPORTANT: use MCTS helper (handles player == -1 flip)
        move = mcts._get_move_from_action(action_id, legal_moves, player=env.current_player)  # type: ignore
        if move is None:
            move = legal_moves[0] if legal_moves else None
        if move is None:
            break

        _, _, done, info = env.step(move)
        if done:
            winner = info["winner"]
            break

    return {"states": states, "players": players, "policies": policies, "winner": winner}
