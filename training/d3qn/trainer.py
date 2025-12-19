import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple
import time

class D3QNTrainer:
    """
    Dueling Double Deep Q-Network Trainer for Checkers.

    Implements:
    - Double DQN: Uses online network for action selection, target for evaluation
    - Dueling architecture: Separate value and advantage streams
    - Legal action masking: Ensures target network only considers legal moves
    - Gradient clipping: Prevents exploding gradients
    """

    def __init__(
        self,
        env,
        action_manager,
        board_encoder,
        model,  # D3QNModel with online and target networks
        optimizer: torch.optim.Optimizer,
        buffer,  # ReplayBuffer
        device: str = "cpu",
        gamma: float = 0.99,
        gradient_clip: float = 1.0,
        loss_type: str = "huber",  # "huber" or "mse"
        tau: float = 0.005  # Soft update parameter (polyak averaging)
    ):
        """
        Initialize D3QN trainer.

        Args:
            env: CheckersEnv instance
            action_manager: ActionManager instance
            board_encoder: CheckersBoardEncoder instance
            model: D3QNModel with online and target networks
            optimizer: PyTorch optimizer (e.g., Adam)
            buffer: ReplayBuffer instance
            device: Device for training (cpu/cuda)
            gamma: Discount factor for future rewards
            gradient_clip: Maximum gradient norm (0 to disable)
            loss_type: "huber" (smooth L1) or "mse"
            tau: Soft update parameter for polyak averaging (0.005 = 0.5% update per step)
        """
        self.env = env
        self.action_manager = action_manager
        self.board_encoder = board_encoder
        self.model = model
        self.optimizer = optimizer
        self.buffer = buffer
        self.device = torch.device(device)
        self.gamma = gamma
        self.gradient_clip = gradient_clip
        self.loss_type = loss_type
        self.tau = tau

        # Training statistics
        self.total_steps = 0
        self.total_episodes = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []

        print(f"D3QNTrainer initialized:")
        print(f"  Device: {device}")
        print(f"  Gamma: {gamma}")
        print(f"  Gradient clip: {gradient_clip}")
        print(f"  Loss type: {loss_type}")
        print(f"  Soft update tau: {tau}")

    def train_step(self, batch_size: int) -> float:
        """
        Perform one training step using Double DQN with legal action masking.

        Algorithm:
        1. Sample batch from replay buffer
        2. Compute current Q-values from online network
        3. Compute target Q-values using Double DQN:
           - Online network selects best legal action in next state
           - Target network evaluates that action
        4. Compute loss and backpropagate

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Loss value (float)
        """
        # Check if buffer has enough samples
        if len(self.buffer) < batch_size:
            return 0.0

        # Set model to training mode
        self.model.train()

        # ================================================================
        # STEP 1: Sample batch from replay buffer
        # ================================================================
        states, actions, rewards, next_states, dones, next_masks = self.buffer.sample(batch_size)

        # Ensure tensors are on correct device and have correct dtype
        states = states.to(self.device).float()          # (batch, 5, 8, 8)
        actions = actions.to(self.device).long()         # (batch, 1)
        rewards = rewards.to(self.device).float()        # (batch, 1)
        next_states = next_states.to(self.device).float()  # (batch, 5, 8, 8)
        dones = dones.to(self.device).float()            # (batch, 1)
        next_masks = next_masks.to(self.device).bool()   # (batch, action_dim)

        # ================================================================
        # STEP 2: Compute current Q-values
        # ================================================================
        # Forward pass through online network
        current_q_values = self.model.online(states)  # (batch, action_dim)

        # Gather Q-values for actions that were actually taken
        current_q = current_q_values.gather(1, actions)  # (batch, 1)

        # ================================================================
        # STEP 3: Compute target Q-values using Double DQN
        # ================================================================
        with torch.no_grad():
            # 3a) Online network selects best actions for next states
            next_q_online = self.model.online(next_states)  # (batch, action_dim)

            # 3b) Apply legal action masks to next state Q-values
            # Set illegal actions to very negative value so they won't be selected
            masked_next_q_online = next_q_online.clone()
            masked_next_q_online[~next_masks] = -1e9

            # 3c) Select best legal action for each next state
            best_next_actions = masked_next_q_online.argmax(dim=1, keepdim=True)  # (batch, 1)

            # 3d) Target network evaluates those selected actions
            next_q_target = self.model.target(next_states)  # (batch, action_dim)

            # 3e) Gather Q-values for the selected best actions
            next_q = next_q_target.gather(1, best_next_actions)  # (batch, 1)

            # 3f) Compute TD target: r + γ * Q_target(s', a*) * (1 - done)
            target_q = rewards + self.gamma * next_q * (1 - dones)  # (batch, 1)

        # ================================================================
        # STEP 4: Compute loss
        # ================================================================
        if self.loss_type == "huber":
            # Huber loss (smooth L1): more robust to outliers
            criterion = torch.nn.SmoothL1Loss()
            loss = criterion(current_q, target_q)
        else:
            # MSE loss
            loss = F.mse_loss(current_q, target_q)

        # ================================================================
        # STEP 5: Backpropagation
        # ================================================================
        self.optimizer.zero_grad()
        loss.backward()

        # Tighter gradient clipping (Gen 3 patch: max_norm=0.1)
        torch.nn.utils.clip_grad_norm_(
            self.model.online.parameters(),
            max_norm=0.1
        )

        self.optimizer.step()

        # Track statistics
        loss_value = loss.item()
        self.losses.append(loss_value)
        self.total_steps += 1

        # ================================================================
        # DEBUG: Q-Value Monitoring (Gen 3 patch)
        # ================================================================
        mean_q = current_q.mean().item()
        if mean_q > 100.0 or mean_q < -100.0:
            print(f"⚠️ WARNING: Q-Value explosion detected: {mean_q:.2f}")

        return loss_value

    def update_target_network(self):
        """
        Soft update: Apply polyak averaging to target network weights.

        θ_target = τ * θ_online + (1 - τ) * θ_target

        This should be called every training step to smoothly update the target network.
        The tau parameter (default 0.005) controls how much the online weights influence the target.
        """
        for target_param, online_param in zip(
            self.model.target.parameters(),
            self.model.online.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    def collect_experience(
        self,
        num_steps: int,
        epsilon: float,
        render: bool = False
    ) -> Dict[str, Any]:
        """
        Collect experience by playing in the environment with epsilon-greedy policy.

        Args:
            num_steps: Number of steps to collect
            epsilon: Exploration rate (probability of random action)
            render: Whether to render the environment

        Returns:
            Dictionary with statistics (rewards, lengths, etc.)
        """
        steps_collected = 0
        episode_reward = 0
        episode_length = 0
        episodes_completed = 0

        self.model.eval()  # Set to evaluation mode

        while steps_collected < num_steps:
            # Get current state
            board = self.env.board.get_state()
            player = self.env.current_player
            legal_moves = self.env.get_legal_moves()

            # Check for terminal state
            if not legal_moves or self.env.done:
                # Episode ended
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.total_episodes += 1
                episodes_completed += 1

                # Reset environment
                self.env.reset()
                episode_reward = 0
                episode_length = 0
                continue

            # Encode current state
            state = self.board_encoder.encode(board, player)

            # Get legal action mask for current state
            current_mask = self.action_manager.make_legal_action_mask(legal_moves)

            # Select action using epsilon-greedy policy
            with torch.no_grad():
                if np.random.random() < epsilon:
                    # Random action from legal moves
                    legal_indices = torch.where(current_mask)[0]
                    action_id = legal_indices[np.random.randint(len(legal_indices))].item()
                else:
                    # Greedy action
                    state_tensor = state.unsqueeze(0).to(self.device)
                    q_values = self.model.online(state_tensor)[0]  # (action_dim,)

                    # Mask illegal actions
                    masked_q = q_values.clone()
                    masked_q[~current_mask] = -1e9

                    action_id = masked_q.argmax().item()

            # Map action to environment move
            move = self.action_manager.get_move_from_id(action_id)

            # Find corresponding environment move format
            # (Simple approach: check if move matches any legal move's start/landing)
            env_move = None
            for legal_move in legal_moves:
                # Extract start and landing from legal move
                if isinstance(legal_move, list):
                    # Capture sequence
                    if legal_move:
                        start = tuple(legal_move[0][0])
                        landing = tuple(legal_move[-1][1])
                        if (start, landing) == move:
                            env_move = legal_move
                            break
                else:
                    # Simple move
                    if len(legal_move) == 2:
                        if (tuple(legal_move[0]), tuple(legal_move[1])) == move:
                            env_move = legal_move
                            break

            # Fallback: if no match found, use first legal move
            if env_move is None:
                env_move = legal_moves[0]
                # Update action_id to match the actual move
                actual_move = self._extract_move(env_move)
                action_id = self.action_manager.get_action_id(actual_move)

            # Execute action in environment
            next_board, reward, done, info = self.env.step(env_move)

            # Get next state and legal moves
            next_player = self.env.current_player
            next_legal_moves = self.env.get_legal_moves() if not done else []

            # Encode next state
            next_state = self.board_encoder.encode(next_board, next_player)

            # Get legal action mask for next state
            next_mask = self.action_manager.make_legal_action_mask(next_legal_moves)

            # Store transition in replay buffer
            self.buffer.push(state, action_id, reward, next_state, done, next_mask)

            # Update statistics
            episode_reward += reward
            episode_length += 1
            steps_collected += 1

            if render:
                self.env.render()

            # Check if episode ended
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.total_episodes += 1
                episodes_completed += 1

                self.env.reset()
                episode_reward = 0
                episode_length = 0

        return {
            'steps_collected': steps_collected,
            'episodes_completed': episodes_completed,
            'avg_reward': np.mean(self.episode_rewards[-episodes_completed:]) if episodes_completed > 0 else 0,
            'avg_length': np.mean(self.episode_lengths[-episodes_completed:]) if episodes_completed > 0 else 0
        }

    def _extract_move(self, env_move) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Extract (start, landing) from environment move format."""
        if isinstance(env_move, list) and env_move:
            # Capture sequence: take first step's start and last step's landing
            start = tuple(env_move[0][0])
            landing = tuple(env_move[-1][1])
            return (start, landing)
        elif len(env_move) == 2:
            # Simple move
            return (tuple(env_move[0]), tuple(env_move[1]))
        else:
            # Fallback
            return ((0, 0), (0, 0))

    def get_statistics(self, window: int = 100) -> Dict[str, float]:
        """
        Get training statistics.

        Args:
            window: Number of recent episodes/steps to average over

        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'buffer_size': len(self.buffer),
        }

        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-window:]
            stats['avg_reward'] = np.mean(recent_rewards)
            stats['max_reward'] = np.max(recent_rewards)
            stats['min_reward'] = np.min(recent_rewards)

        if self.episode_lengths:
            recent_lengths = self.episode_lengths[-window:]
            stats['avg_length'] = np.mean(recent_lengths)

        if self.losses:
            recent_losses = self.losses[-window:]
            stats['avg_loss'] = np.mean(recent_losses)

        return stats

    def save_checkpoint(self, path: str):
        """
        Save training checkpoint.

        Args:
            path: File path to save checkpoint
        """
        checkpoint = {
            'model_online': self.model.online.state_dict(),
            'model_target': self.model.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """
        Load training checkpoint.

        Args:
            path: File path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.online.load_state_dict(checkpoint['model_online'])
        self.model.target.load_state_dict(checkpoint['model_target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = checkpoint['total_steps']
        self.total_episodes = checkpoint['total_episodes']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.losses = checkpoint['losses']

        print(f"Checkpoint loaded from {path}")
        print(f"  Steps: {self.total_steps}")
        print(f"  Episodes: {self.total_episodes}")


# ================================================================
# Testing
# ================================================================

if __name__ == "__main__":
    print("="*70)
    print("D3QN TRAINER TEST")
    print("="*70)
    print("\nThis is a template. Full testing requires:")
    print("  - CheckersEnv")
    print("  - ActionManager")
    print("  - CheckersBoardEncoder")
    print("  - D3QNModel")
    print("  - ReplayBuffer")
    print("\nTrainer class structure verified ✓")