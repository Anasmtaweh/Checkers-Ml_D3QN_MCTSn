#!/usr/bin/env python3
"""
main.py - D3QN Checkers Training Entry Point (GENERATION 7)

Trains a Dueling Double Deep Q-Network agent to play Checkers through
self-play against a random opponent.

**GENERATION 7 CHANGES (Critical Reward Restructuring):**
- REWARD_SCALE changed from 100.0 → 1.0 (eliminates artificial shrinking)
- Win reward: +1.0 (was +100 → scaled to 0.01, now properly 1.0)
- Loss penalty: -1.0 (was -75 → scaled to -0.75, now symmetric -1.0)
- Multi-Jump Capture: +0.01 (was +20 → scaled to 0.2, now minimal 0.01)
- Single Capture: +0.001 (was +5 → scaled to 0.05, now minimal 0.001)
- Living Tax: -0.0001 (microscopic negative for non-capture moves)

**WHY THIS FIXES REWARD HACKING:**
Old Gen 6 loophole: 4 captures (+80) + loss (-75) = +5 profit
New Gen 7 math: 4 captures (+0.04) + loss (-1.0) = -0.96 ← LOSING IS BAD!

Features:
- Resume training from checkpoints
- Automatic epsilon adjustment on resume
- Episode tracking
- Fixed reward specification (no more exploit)

Author: ML Engineer
Date: December 19, 2025 (Gen 7 Release)
"""

import torch
import torch.optim as optim
import numpy as np
import os
import re
from pathlib import Path
from typing import Optional, Tuple, Any

# Import project modules
from checkers_env.env import CheckersEnv
from checkers_agents.random_agent import CheckersRandomAgent
from training.common.action_manager import ActionManager
from training.common.board_encoder import CheckersBoardEncoder
from training.common.buffer import ReplayBuffer
from training.d3qn.model import D3QNModel
from training.d3qn.trainer import D3QNTrainer


# ════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS / CONFIGURATION (GENERATION 7)
# ════════════════════════════════════════════════════════════════════

NUM_EPISODES = 15000
BATCH_SIZE = 128  # Increased for more stable gradients (Gen 2 patch)
GAMMA = 0.99
LEARNING_RATE = 2e-5  # Increased for faster learning (Gen 4 patch)

# Epsilon-Greedy Exploration
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 5000  # Episodes to decay from start to end

# Soft Updates (Gen 2 patch)
# Target network uses polyak averaging (tau=0.005) every step

# Replay Buffer
BUFFER_CAPACITY = 100000
MIN_BUFFER_SIZE = 3000  # Start training after this many transitions

# Logging and Saving
LOG_FREQ = 10  # Log every N episodes
SAVE_FREQ = 500  # Save checkpoint every N episodes

# Paths
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Resume Configuration
# Set to checkpoint path to resume training, or None to start fresh
# Example: "checkpoints/model_episode_2500.pth"
RESUME_CHECKPOINT = None  # Start Fresh (Gen 7)


# ════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════

def calculate_epsilon(episode, epsilon_start, epsilon_end, epsilon_decay):
    """
    Calculate epsilon for current episode using linear decay.

    Args:
        episode: Current episode number
        epsilon_start: Initial epsilon value
        epsilon_end: Final epsilon value
        epsilon_decay: Number of episodes to decay over

    Returns:
        Current epsilon value
    """
    epsilon = epsilon_start - (epsilon_start - epsilon_end) * (episode / epsilon_decay)
    return max(epsilon_end, epsilon)


def select_agent_action(model, state, legal_moves, action_manager, epsilon, device) -> Tuple[Any, Optional[int], torch.Tensor]:
    """
    Select action using epsilon-greedy policy.

    Args:
        model: D3QNModel (uses online network)
        state: Encoded state tensor (5, 8, 8)
        legal_moves: List of legal moves from environment
        action_manager: ActionManager instance
        epsilon: Current exploration rate
        device: Device for computation

    Returns:
        Tuple of (env_move, action_id, legal_mask)
    """
    # Get legal action mask
    legal_mask = action_manager.make_legal_action_mask(legal_moves)
    legal_indices = torch.where(legal_mask)[0]

    if len(legal_indices) == 0:
        return None, None, legal_mask

    # Epsilon-greedy selection
    if np.random.random() < epsilon:
        # Random action from legal moves
        action_id = int(legal_indices[np.random.randint(len(legal_indices))].item())
    else:
        # Greedy action
        model.eval()
        with torch.no_grad():
            state_tensor = state.unsqueeze(0).to(device)
            q_values = model.get_q_values(state_tensor)[0]

            # Apply legal mask
            masked_q = q_values.clone()
            masked_q[~legal_mask] = -1e9

            action_id = int(masked_q.argmax().item())

    # Map action_id to environment move
    move = action_manager.get_move_from_id(action_id)

    # Find matching environment move
    env_move = None
    for legal_move in legal_moves:
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

    # Fallback: use first legal move if no match
    if env_move is None:
        env_move = legal_moves[0]

    return env_move, action_id, legal_mask


# ════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ════════════════════════════════════════════════════════════════════

def main():
    """Main training loop."""

    print("="*70)
    print("D3QN CHECKERS TRAINING - GENERATION 7 (Reward Fix)")
    print("="*70)

    # ════════════════════════════════════════════════════════════════
    # DEVICE DETECTION
    # ════════════════════════════════════════════════════════════════
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # ════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ════════════════════════════════════════════════════════════════
    print("\nInitializing components...")

    # Environment
    env = CheckersEnv()
    print("  ✓ CheckersEnv")

    # Action Manager
    action_manager = ActionManager(device=device)
    print(f"  ✓ ActionManager ({action_manager.action_dim} actions)")

    # Board Encoder
    encoder = CheckersBoardEncoder()
    print("  ✓ CheckersBoardEncoder")

    # D3QN Model
    model = D3QNModel(
        action_dim=action_manager.action_dim,
        device=device
    )
    print(f"  ✓ D3QNModel")

    # ════════════════════════════════════════════════════════════════
    # CHECKPOINT LOADING (RESUME FUNCTIONALITY)
    # ════════════════════════════════════════════════════════════════
    start_episode = 1
    epsilon_start = EPSILON_START
    resumed_step_count = 0
    resumed_rewards = []
    resumed_lengths = []
    resumed_losses = []

    if RESUME_CHECKPOINT is not None and os.path.exists(RESUME_CHECKPOINT):
        print(f"\n🔄 Resuming training from: {RESUME_CHECKPOINT}")

        try:
            # Load checkpoint
            checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)

            # Load model weights
            if isinstance(checkpoint, dict):
                if "model_online" in checkpoint:
                    model.online.load_state_dict(checkpoint["model_online"])
                    model.target.load_state_dict(checkpoint["model_target"])
                elif "online_model_state_dict" in checkpoint:
                    model.online.load_state_dict(checkpoint["online_model_state_dict"])
                    model.target.load_state_dict(checkpoint["target_model_state_dict"])
                else:
                    print("  ⚠️ Unknown checkpoint format, loading as state dict")
                    model.online.load_state_dict(checkpoint)
                    model.target.load_state_dict(checkpoint)
            else:
                # Direct state dict
                model.online.load_state_dict(checkpoint)
                model.target.load_state_dict(checkpoint)

            # Extract episode number from filename
            match = re.search(r"episode_(\d+)", RESUME_CHECKPOINT)
            if match:
                start_episode = int(match.group(1)) + 1

                # Adjust epsilon based on progress
                decay_progress = min(1.0, start_episode / EPSILON_DECAY)
                epsilon_start = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * decay_progress)

                print(f"  ✓ Loaded model weights")
                print(f"  ✓ Starting from episode: {start_episode}")
                print(f"  ✓ Adjusted epsilon start: {epsilon_start:.4f}")

                # Try to load training statistics if available
                if isinstance(checkpoint, dict):
                    if "step_count" in checkpoint:
                        resumed_step_count = checkpoint.get("step_count", 0)
                        print(f"  ✓ Resumed step count: {resumed_step_count}")
                    if "episode_rewards" in checkpoint:
                        resumed_rewards = checkpoint.get("episode_rewards", [])
                        print(f"  ✓ Loaded {len(resumed_rewards)} episode rewards")
                    if "episode_lengths" in checkpoint:
                        resumed_lengths = checkpoint.get("episode_lengths", [])
                    if "losses" in checkpoint:
                        resumed_losses = checkpoint.get("losses", [])
            else:
                print("  ⚠️ Could not parse episode number from filename")
                print("  ⚠️ Starting from episode 1")

        except Exception as e:
            print(f"  ❌ Error loading checkpoint: {e}")
            print("  ⚠️ Starting fresh training")
            start_episode = 1
            epsilon_start = EPSILON_START
    else:
        if RESUME_CHECKPOINT is not None:
            print(f"\n⚠️ Checkpoint not found: {RESUME_CHECKPOINT}")
        print("✨ Starting fresh training run (Generation 7)")

    # ════════════════════════════════════════════════════════════════
    # OPTIMIZER (Initialize after loading model)
    # ════════════════════════════════════════════════════════════════
    optimizer = optim.Adam(model.online.parameters(), lr=LEARNING_RATE)

    # Try to load optimizer state if resuming
    if RESUME_CHECKPOINT is not None and os.path.exists(RESUME_CHECKPOINT):
        try:
            checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)
            if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                print("  ✓ Loaded optimizer state")
        except Exception as e:
            print(f"  ⚠️ Could not load optimizer state: {e}")

    print("  ✓ Adam Optimizer")

    # Replay Buffer
    buffer = ReplayBuffer(
        capacity=BUFFER_CAPACITY,
        action_dim=action_manager.action_dim,
        device=device
    )
    print(f"  ✓ ReplayBuffer ({BUFFER_CAPACITY:,} capacity)")

    # Trainer
    trainer = D3QNTrainer(
        env=env,
        action_manager=action_manager,
        board_encoder=encoder,
        model=model,
        optimizer=optimizer,
        buffer=buffer,
        device=device,
        gamma=GAMMA,
        gradient_clip=1.0,
        loss_type="huber",
        tau=0.005  # Soft update parameter (Gen 2 patch)
    )
    print("  ✓ D3QNTrainer")

    # Opponent (Random Agent)
    opponent = CheckersRandomAgent()
    print("  ✓ RandomAgent (Opponent)")

    print("\n" + "="*70)
    print("TRAINING START - GENERATION 7")
    print("="*70)
    print(f"Episodes: {start_episode} → {NUM_EPISODES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Gamma: {GAMMA}")
    print(f"Epsilon: {epsilon_start:.4f} → {EPSILON_END} (over {EPSILON_DECAY} episodes)")
    print(f"Soft updates: tau=0.005 (every step)")
    print("\n🔧 Gen 7 Reward Structure:")
    print("  • Win: +1.0 (was +100→0.01, now proper 1.0)")
    print("  • Loss: -1.0 (was -75→-0.75, now symmetric -1.0)")
    print("  • Multi-Jump: +0.01 (was +20→0.2, now minimal 0.01)")
    print("  • Single Capture: +0.001 (was +5→0.05, now minimal 0.001)")
    print("  • Living Tax: -0.0001 (microscopic negative)")
    print("\n✓ Loophole closed: 4 captures (+0.04) + loss (-1.0) = -0.96")
    print("="*70 + "\n")

    # ════════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ════════════════════════════════════════════════════════════════

    step_count = resumed_step_count
    episode_rewards = resumed_rewards.copy()
    episode_lengths = resumed_lengths.copy()
    losses = resumed_losses.copy()

    for episode in range(start_episode, NUM_EPISODES + 1):
        # Calculate epsilon for this episode (using adjusted start)
        epsilon = calculate_epsilon(episode, epsilon_start, EPSILON_END, EPSILON_DECAY)

        # Reset environment
        state = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        info = {}

        # ════════════════════════════════════════════════════════════
        # GAME LOOP
        # ════════════════════════════════════════════════════════════
        while not done:
            current_player = env.current_player
            legal_moves = env.get_legal_moves()

            # Check for terminal state
            if not legal_moves:
                done = True
                break

            # ════════════════════════════════════════════════════════
            # AGENT'S TURN (Player 1)
            # ════════════════════════════════════════════════════════
            if current_player == 1:
                # Encode current state
                encoded_state = encoder.encode(state, player=current_player)

                # Select action using epsilon-greedy
                env_move, action_id, current_legal_mask = select_agent_action(
                    model, encoded_state, legal_moves, action_manager, epsilon, device
                )

                if env_move is None:
                    done = True
                    break

                # Execute action in environment
                next_state, reward, done, info = env.step(env_move)

                # Get next state's legal moves (for mask)
                next_legal_moves = env.get_legal_moves() if not done else []
                next_legal_mask = action_manager.make_legal_action_mask(next_legal_moves)

                # Encode next state
                next_encoded_state = encoder.encode(next_state, player=env.current_player)

                # Store transition in replay buffer
                if action_id is not None:
                    buffer.push(
                        state=encoded_state,
                        action=int(action_id),
                        reward=reward,
                        next_state=next_encoded_state,
                        done=done,
                        next_legal_mask=next_legal_mask
                    )

                # Train if buffer has enough samples
                if len(buffer) >= MIN_BUFFER_SIZE:
                    model.train()
                    loss = trainer.train_step(BATCH_SIZE)
                    losses.append(loss)
                    step_count += 1

                    # Soft update target network every step (Gen 2 patch)
                    trainer.update_target_network()

                # Update tracking
                episode_reward += reward
                episode_length += 1
                state = next_state

            # ════════════════════════════════════════════════════════
            # OPPONENT'S TURN (Player 2)
            # ════════════════════════════════════════════════════════
            else:
                # Opponent selects random action
                opponent_move = opponent.select_action(env)

                if opponent_move is None:
                    done = True
                    break

                # Execute opponent's move
                next_state, reward, done, info = env.step(opponent_move)

                # Note: We do NOT store opponent's transitions in buffer
                # The agent only learns from its own experiences

                episode_length += 1
                state = next_state

        # ════════════════════════════════════════════════════════════
        # END OF EPISODE
        # ════════════════════════════════════════════════════════════
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # ════════════════════════════════════════════════════════════
        # LOGGING
        # ════════════════════════════════════════════════════════════
        if episode % LOG_FREQ == 0:
            avg_loss = np.mean(losses[-100:]) if losses else 0.0

            # Check environment's official winner record
            winner = info.get('winner', 0)

            if winner == 1:
                outcome = "WIN "
            elif winner == -1:
                outcome = "LOSS"
            else:
                outcome = "DRAW"

            # Print the truth
            log_msg = f"Episode {episode:5d} {outcome} Reward {episode_reward:7.2f} Length {episode_length:5.1f} Loss {avg_loss:.2e} Epsilon {epsilon:.4f}"
            print(log_msg)

            # Write to log file
            with open("training_log.txt", "a") as f:
                f.write(log_msg + "\n")

        # ════════════════════════════════════════════════════════════
        # CHECKPOINTING
        # ════════════════════════════════════════════════════════════
        if episode % SAVE_FREQ == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_episode_{episode}.pth")
            torch.save({
                'episode': episode,
                'model_online': model.online.state_dict(),
                'model_target': model.target.state_dict(),
                'optimizer': optimizer.state_dict(),
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'losses': losses,
                'step_count': step_count,
                'generation': 7,  # Mark as Gen 7
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")

    # ════════════════════════════════════════════════════════════════
    # TRAINING COMPLETE
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("TRAINING COMPLETE - GENERATION 7")
    print("="*70)

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "model_final_gen7.pth")
    torch.save({
        'episode': NUM_EPISODES,
        'model_online': model.online.state_dict(),
        'model_target': model.target.state_dict(),
        'optimizer': optimizer.state_dict(),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'losses': losses,
        'step_count': step_count,
        'generation': 7,
    }, final_path)
    print(f"Final model saved: {final_path}")

    # Print statistics
    print(f"\nTraining Statistics:")
    print(f"  Total episodes: {NUM_EPISODES}")
    print(f"  Total steps: {step_count}")
    print(f"  Final avg reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"  Final avg length (last 100): {np.mean(episode_lengths[-100:]):.1f}")
    print(f"  Final avg loss (last 100): {np.mean(losses[-100:]):.4f}" if losses else "")
    print("="*70)


# ════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("\nStarting D3QN Checkers Training (Generation 7)...")
    print("Press Ctrl+C to interrupt training\n")

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Progress has been saved in checkpoints/")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)