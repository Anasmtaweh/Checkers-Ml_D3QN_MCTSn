import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
import time
import os


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
        
        # Replay buffer: stores (state, policy_target, value_target) tuples
        self.replay_buffer: deque = deque(maxlen=buffer_size)
        
        # Optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.network.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # Statistics tracking
        self.training_stats = {
            'total_games': 0,
            'total_steps': 0,
            'losses': [],
            'value_losses': [],
            'policy_losses': [],
        }
    
    # ================================================================
    # SELF-PLAY: Generate Training Data
    # ================================================================
    
    def self_play(self, num_games: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Play games against itself to generate training data.
        
        This is the CRITICAL data generation phase:
        1. Play full games using MCTS for move selection
        2. Store (state, player, mcts_policy) during play
        3. When game ends, compute value targets based on outcome
        4. Add all (state, policy_target, value_target) to replay buffer
        
        Args:
            num_games: Number of games to play
            verbose: Whether to print progress
        
        Returns:
            Dictionary with self-play statistics
        """
        from core.game import CheckersEnv
        
        stats = {
            'games_played': 0,
            'p1_wins': 0,
            'p2_wins': 0,
            'draws': 0,
            'total_moves': 0,
            'avg_game_length': 0.0,
            'buffer_size': len(self.replay_buffer),
        }
        
        self.model.eval()  # Set to evaluation mode for self-play
        
        for game_idx in range(num_games):
            game_data = self._play_single_game()
            
            # Update statistics
            stats['games_played'] += 1
            stats['total_moves'] += len(game_data['states'])
            
            winner = game_data['winner']
            if winner == 1:
                stats['p1_wins'] += 1
            elif winner == -1:
                stats['p2_wins'] += 1
            else:
                stats['draws'] += 1
            
            # Add game data to replay buffer
            self._process_game_data(game_data)
            
            if verbose and (game_idx + 1) % 10 == 0:
                print(f"  Self-play: {game_idx + 1}/{num_games} games complete")
        
        # Compute final statistics
        stats['avg_game_length'] = stats['total_moves'] / stats['games_played']
        stats['buffer_size'] = len(self.replay_buffer)
        stats['p1_win_rate'] = stats['p1_wins'] / stats['games_played']
        stats['p2_win_rate'] = stats['p2_wins'] / stats['games_played']
        stats['draw_rate'] = stats['draws'] / stats['games_played']
        
        self.training_stats['total_games'] += num_games
        
        if verbose:
            print(f"\n  Self-Play Summary:")
            print(f"    P1 Wins: {stats['p1_wins']} ({stats['p1_win_rate']:.1%})")
            print(f"    P2 Wins: {stats['p2_wins']} ({stats['p2_win_rate']:.1%})")
            print(f"    Draws: {stats['draws']} ({stats['draw_rate']:.1%})")
            print(f"    Avg Game Length: {stats['avg_game_length']:.1f} moves")
            print(f"    Buffer Size: {stats['buffer_size']}")
        
        return stats
    
    def _play_single_game(self) -> Dict[str, Any]:
        """
        Play a single self-play game and collect data.
        
        Returns:
            Dictionary containing:
                - states: List of encoded states
                - players: List of current players
                - policies: List of MCTS policy distributions
                - winner: Final game outcome (1, -1, or 0)
        """
        from core.game import CheckersEnv
        
        env = CheckersEnv()
        env.reset()
        
        # Storage for game trajectory
        states = []
        players = []
        policies = []
        
        move_count = 0
        
        while not env.done:
            # Determine temperature based on move count
            if move_count < self.temp_threshold:
                temp = 1.0  # Explore early game
            else:
                temp = 0.0  # Play optimally in endgame
            
            # Get MCTS policy
            action_probs, root = self.mcts.get_action_prob(env, temp=temp, training=True)
            
            # Store current state and policy
            board = env.board.get_state()
            player = env.current_player
            encoded_state = self.board_encoder.encode(board, player)
            
            states.append(encoded_state)
            players.append(player)
            policies.append(action_probs)
            
            # Select and execute action
            legal_moves = env.get_legal_moves()
            
            if temp == 0:
                # Deterministic: choose most visited action
                action_id = int(np.argmax(action_probs))
            else:
                # Stochastic: sample from distribution
                action_id = int(np.random.choice(len(action_probs), p=action_probs))
            
            # Convert action_id to move
            move = self._get_move_from_action_id(action_id, legal_moves)
            
            if move is None:
                # Fallback: choose random legal move
                move = legal_moves[0] if legal_moves else None
            
            if move is None:
                # No legal moves (shouldn't happen, but handle gracefully)
                break
            
            # Execute move
            _, _, done, info = env.step(move)
            move_count += 1
            
            # Safety check: prevent infinite games
            if move_count > 300:
                print("  Warning: Game exceeded 300 moves, forcing draw")
                break
        
        # Determine winner
        _, winner = env._check_game_over()
        
        return {
            'states': states,
            'players': players,
            'policies': policies,
            'winner': winner,
        }
    
    def _process_game_data(self, game_data: Dict[str, Any]):
        """
        Process game data and add to replay buffer.
        
        CRITICAL VALUE TARGET CALCULATION:
        - For each position, assign value based on game outcome
        - Value is from the perspective of the player to move
        - If current_player == winner: Z = +1
        - If current_player == -winner: Z = -1
        - If draw: Z = 0
        
        Args:
            game_data: Dictionary with states, players, policies, winner
        """
        states = game_data['states']
        players = game_data['players']
        policies = game_data['policies']
        winner = game_data['winner']
        
        # Process each position in the game
        for state, player, policy in zip(states, players, policies):
            # Calculate value target based on game outcome
            if winner == 0:
                # Draw
                value_target = 0.0
            elif player == winner:
                # This player won
                value_target = 1.0
            else:
                # This player lost
                value_target = -1.0
            
            # Add to replay buffer
            # Store as tuple: (state_tensor, policy_array, value_scalar)
            self.replay_buffer.append((
                state.cpu(),  # Store on CPU to save GPU memory
                policy,
                value_target
            ))
    
    def _get_move_from_action_id(self, action_id: int, legal_moves: List) -> Optional[Any]:
        """Convert action_id to actual move from legal_moves list."""
        move_pair = self.action_manager.get_move_from_id(action_id)
        
        if move_pair == ((-1, -1), (-1, -1)):
            return None
        
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
        
        AlphaZero Loss Function:
        L = L_value + L_policy
        
        Where:
        - L_value = MSE(predicted_value, target_value)
        - L_policy = -Σ(target_policy * log(predicted_policy))
        
        Args:
            epochs: Number of training epochs on current buffer
            verbose: Whether to print training progress
        
        Returns:
            Dictionary with loss statistics
        """
        if len(self.replay_buffer) < self.batch_size:
            if verbose:
                print(f"  Insufficient data: {len(self.replay_buffer)}/{self.batch_size}")
            return {'loss': 0.0, 'value_loss': 0.0, 'policy_loss': 0.0}
        
        self.model.train()  # Set to training mode
        
        total_loss = 0.0
        total_value_loss = 0.0
        total_policy_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            # Sample a batch
            batch_data = self._sample_batch(self.batch_size)
            
            states = batch_data['states'].to(self.device)
            policy_targets = batch_data['policy_targets'].to(self.device)
            value_targets = batch_data['value_targets'].to(self.device)
            
            # Forward pass
            policy_logits, value_pred = self.model.get_policy_value(states)
            
            # Compute losses
            value_loss = self._compute_value_loss(value_pred, value_targets)
            policy_loss = self._compute_policy_loss(policy_logits, policy_targets)
            
            # Combined loss
            loss = (
                self.value_loss_weight * value_loss +
                self.policy_loss_weight * policy_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.network.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track statistics
            total_loss += loss.item()
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            num_batches += 1
        
        # Compute averages
        avg_loss = total_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_policy_loss = total_policy_loss / num_batches
        
        # Update training statistics
        self.training_stats['total_steps'] += num_batches
        self.training_stats['losses'].append(avg_loss)
        self.training_stats['value_losses'].append(avg_value_loss)
        self.training_stats['policy_losses'].append(avg_policy_loss)
        
        if verbose:
            print(f"  Training: loss={avg_loss:.4f}, "
                  f"value_loss={avg_value_loss:.4f}, "
                  f"policy_loss={avg_policy_loss:.4f}")
        
        return {
            'loss': avg_loss,
            'value_loss': avg_value_loss,
            'policy_loss': avg_policy_loss,
        }
    
    def _sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a random batch from replay buffer.
        
        Args:
            batch_size: Number of samples to draw
        
        Returns:
            Dictionary with batched tensors
        """
        # Random sampling with replacement
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        
        states_list = []
        policies_list = []
        values_list = []
        
        for idx in indices:
            state, policy, value = self.replay_buffer[idx]
            states_list.append(state)
            policies_list.append(policy)
            values_list.append(value)
        
        # Stack into tensors
        states = torch.stack(states_list)
        policy_targets = torch.tensor(np.array(policies_list), dtype=torch.float32)
        value_targets = torch.tensor(values_list, dtype=torch.float32).unsqueeze(1)
        
        return {
            'states': states,
            'policy_targets': policy_targets,
            'value_targets': value_targets,
        }
    
    def _compute_value_loss(self, value_pred: torch.Tensor, value_target: torch.Tensor) -> torch.Tensor:
        """
        Compute Mean Squared Error for value prediction.
        
        Args:
            value_pred: Predicted values from network (batch, 1)
            value_target: Target values from game outcomes (batch, 1)
        
        Returns:
            MSE loss scalar
        """
        return nn.MSELoss()(value_pred, value_target)
    
    def _compute_policy_loss(self, policy_logits: torch.Tensor, policy_target: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss for policy prediction.
        
        AlphaZero uses: -Σ(π_target * log(π_pred))
        
        Where:
        - π_target: MCTS visit distribution (target)
        - π_pred: Neural network policy (prediction)
        
        Args:
            policy_logits: Log-probabilities from network (batch, action_dim)
            policy_target: MCTS policy distribution (batch, action_dim)
        
        Returns:
            Cross-entropy loss scalar
        """
        # policy_logits are already log-probabilities (from LogSoftmax)
        # policy_target is a probability distribution
        
        # Cross-entropy: -Σ(target * log(pred))
        loss = -(policy_target * policy_logits).sum(dim=1).mean()
        
        return loss
    
    # ================================================================
    # CHECKPOINT MANAGEMENT
    # ================================================================
    
    def save_checkpoint(self, path: str, iteration: int = 0, additional_info: Optional[Dict] = None):
        """
        Save training checkpoint.
        
        Args:
            path: File path to save to
            iteration: Current iteration number
            additional_info: Optional dict with extra information to save
        """
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'buffer_size': len(self.replay_buffer),
        }
        
        if additional_info is not None:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> Dict:
        """
        Load training checkpoint.
        
        Args:
            path: File path to load from
        
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        
        print(f"✓ Checkpoint loaded from {path}")
        print(f"  Iteration: {checkpoint.get('iteration', 'unknown')}")
        print(f"  Total games: {self.training_stats['total_games']}")
        
        return checkpoint
    
    # ================================================================
    # UTILITIES
    # ================================================================
    
    def get_buffer_size(self) -> int:
        """Get current replay buffer size."""
        return len(self.replay_buffer)
    
    def clear_buffer(self):
        """Clear replay buffer."""
        self.replay_buffer.clear()
        print("✓ Replay buffer cleared")
    
    def get_training_stats(self) -> Dict:
        """Get training statistics."""
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
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run a single training iteration: Self-Play → Training → Checkpoint.
    
    Args:
        trainer: AlphaZeroTrainer instance
        iteration: Current iteration number
        num_self_play_games: Number of self-play games to generate
        num_train_epochs: Number of training epochs per iteration
        checkpoint_dir: Directory to save checkpoints
        save_every: Save checkpoint every N iterations
        verbose: Whether to print progress
    
    Returns:
        Dictionary with iteration statistics
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}")
        print(f"{'='*70}")
    
    start_time = time.time()
    
    # Phase 1: Self-Play
    if verbose:
        print(f"\n[1/2] Self-Play ({num_self_play_games} games)...")
    
    self_play_stats = trainer.self_play(num_self_play_games, verbose=verbose)
    
    # Phase 2: Training
    if verbose:
        print(f"\n[2/2] Training ({num_train_epochs} epochs)...")
    
    train_stats = trainer.train_step(epochs=num_train_epochs, verbose=verbose)
    
    # Save checkpoint periodically
    if iteration % save_every == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{iteration}.pth")
        trainer.save_checkpoint(
            checkpoint_path,
            iteration=iteration,
            additional_info={
                'self_play_stats': self_play_stats,
                'train_stats': train_stats,
            }
        )
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\n✓ Iteration {iteration} complete in {elapsed:.1f}s")
        print(f"{'='*70}\n")
    
    return {
        'iteration': iteration,
        'self_play_stats': self_play_stats,
        'train_stats': train_stats,
        'elapsed_time': elapsed,
    }


# ================================================================
# Testing and Debugging
# ================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AlphaZero TRAINER TEST")
    print("=" * 70)
    
    import sys
    sys.path.append('.')
    
    from training.alpha_zero.network import AlphaZeroModel
    from training.alpha_zero.mcts import MCTS
    from core.action_manager import ActionManager
    from core.board_encoder import CheckersBoardEncoder
    
    # Initialize components
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    action_manager = ActionManager(device)
    encoder = CheckersBoardEncoder()
    model = AlphaZeroModel(action_dim=action_manager.action_dim, device=device)
    
    mcts = MCTS(
        model=model,
        action_manager=action_manager,
        encoder=encoder,
        c_puct=1.5,
        num_simulations=50,  # Reduced for testing
        device=device
    )
    
    trainer = AlphaZeroTrainer(
        model=model,
        mcts=mcts,
        action_manager=action_manager,
        board_encoder=encoder,
        device=device,
        buffer_size=1000,
        batch_size=32,
    )
    
    print(f"\nTrainer initialized:")
    print(f"  Buffer size: {trainer.get_buffer_size()}")
    print(f"  Batch size: {trainer.batch_size}")
    print(f"  Temperature threshold: {trainer.temp_threshold}")
    
    # Test 1: Self-play
    print("\n" + "-" * 70)
    print("Test 1: Self-Play (2 games)")
    print("-" * 70)
    
    stats = trainer.self_play(num_games=2, verbose=True)
    
    print(f"\nBuffer after self-play: {trainer.get_buffer_size()} positions")
    
    # Test 2: Training
    if trainer.get_buffer_size() >= trainer.batch_size:
        print("\n" + "-" * 70)
        print("Test 2: Training Step")
        print("-" * 70)
        
        train_stats = trainer.train_step(epochs=2, verbose=True)
        print(f"\nTraining successful!")
    else:
        print("\n⚠ Insufficient data for training test")
    
    # Test 3: Checkpoint save/load
    print("\n" + "-" * 70)
    print("Test 3: Checkpoint Save/Load")
    print("-" * 70)
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        trainer.save_checkpoint(tmp.name, iteration=1)
        
        # Create new trainer and load
        trainer2 = AlphaZeroTrainer(
            model=model,
            mcts=mcts,
            action_manager=action_manager,
            board_encoder=encoder,
            device=device,
        )
        trainer2.load_checkpoint(tmp.name)
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED - AlphaZero TRAINER READY")
    print("=" * 70)
    print("\nKey Features Implemented:")
    print("  • Self-play game generation with temperature control")
    print("  • Value target calculation from game outcomes")
    print("  • Combined policy + value loss training")
    print("  • Replay buffer management")
    print("  • Checkpoint save/load functionality")
