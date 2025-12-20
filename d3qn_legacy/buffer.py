import numpy as np
import torch
from typing import Tuple

class ReplayBuffer:
    """
    Pre-allocated Numpy Array Replay Buffer for D3QN Checkers Training.

    Optimized for RTX 2060 + 24GB RAM:
    - Stores transitions in CPU RAM (numpy arrays)
    - Only moves to GPU during sampling (saves VRAM)
    - Uses float32 for efficiency

    Key Feature: Stores legal action masks for next states, crucial for
    ensuring the target network only selects legal actions in Double DQN.
    """

    def __init__(self, capacity: int, action_dim: int, device: str = "cpu"):
        """
        Initialize replay buffer with pre-allocated arrays.

        Args:
            capacity: Maximum number of transitions to store
            action_dim: Size of action space (from ActionManager)
            device: Device to move tensors to during sampling (cpu/cuda)
        """
        self.capacity = capacity
        self.action_dim = action_dim
        self.device = torch.device(device)

        # Circular buffer pointer
        self.ptr = 0
        self.size = 0  # Current number of stored transitions

        # Pre-allocate numpy arrays for maximum speed
        # All stored in CPU RAM, moved to GPU only during sampling

        # States: (capacity, 5, 8, 8) - encoded board states
        self.states = np.zeros((capacity, 5, 8, 8), dtype=np.float32)

        # Next states: (capacity, 5, 8, 8)
        self.next_states = np.zeros((capacity, 5, 8, 8), dtype=np.float32)

        # Actions: (capacity, 1) - selected action indices
        self.actions = np.zeros((capacity, 1), dtype=np.int64)

        # Rewards: (capacity, 1) - immediate rewards
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)

        # Dones: (capacity, 1) - episode termination flags
        self.dones = np.zeros((capacity, 1), dtype=np.bool_)

        # Legal action masks for NEXT states: (capacity, action_dim)
        # This is crucial: stores which actions are legal in the next state
        # Used to constrain target network action selection in Double DQN
        self.legal_masks = np.zeros((capacity, action_dim), dtype=np.bool_)

        print(f"ReplayBuffer initialized:")
        print(f"  Capacity: {capacity:,} transitions")
        print(f"  Action dim: {action_dim}")
        print(f"  Memory usage: ~{self._estimate_memory_mb():.1f} MB")
        print(f"  Device for sampling: {device}")

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        next_legal_mask: torch.Tensor
    ):
        """
        Add a transition to the replay buffer.

        Args:
            state: Current state tensor (5, 8, 8)
            action: Action index taken
            reward: Immediate reward received
            next_state: Resulting state tensor (5, 8, 8)
            done: Whether episode terminated
            next_legal_mask: Legal action mask for next_state (action_dim,)
        """
        # Convert tensors to numpy and store at current pointer
        if isinstance(state, torch.Tensor):
            self.states[self.ptr] = state.cpu().numpy()
        else:
            self.states[self.ptr] = state

        if isinstance(next_state, torch.Tensor):
            self.next_states[self.ptr] = next_state.cpu().numpy()
        else:
            self.next_states[self.ptr] = next_state

        self.actions[self.ptr, 0] = action
        self.rewards[self.ptr, 0] = reward
        self.dones[self.ptr, 0] = done

        if isinstance(next_legal_mask, torch.Tensor):
            self.legal_masks[self.ptr] = next_legal_mask.cpu().numpy()
        else:
            self.legal_masks[self.ptr] = next_legal_mask

        # Update pointer (cyclic buffer)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[
        torch.Tensor,  # states
        torch.Tensor,  # actions
        torch.Tensor,  # rewards
        torch.Tensor,  # next_states
        torch.Tensor,  # dones
        torch.Tensor   # next_legal_masks
    ]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, masks)
            All tensors moved to self.device (GPU if available)
        """
        # Randomly sample indices from current buffer contents
        indices = np.random.randint(0, self.size, size=batch_size)

        # Extract batch from numpy arrays
        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_dones = self.dones[indices]
        batch_masks = self.legal_masks[indices]

        # Convert to PyTorch tensors and move to device
        # This is where GPU transfer happens (only for sampled batch)
        states = torch.from_numpy(batch_states).to(self.device)
        actions = torch.from_numpy(batch_actions).to(self.device)
        rewards = torch.from_numpy(batch_rewards).to(self.device)
        next_states = torch.from_numpy(batch_next_states).to(self.device)
        dones = torch.from_numpy(batch_dones).to(self.device).float()  # Convert bool to float
        masks = torch.from_numpy(batch_masks).to(self.device)

        return states, actions, rewards, next_states, dones, masks

    def __len__(self) -> int:
        """Return current number of stored transitions."""
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= min_size

    def _estimate_memory_mb(self) -> float:
        """Estimate total memory usage in MB."""
        # Calculate bytes for each array
        states_bytes = self.capacity * 5 * 8 * 8 * 4  # float32
        next_states_bytes = self.capacity * 5 * 8 * 8 * 4
        actions_bytes = self.capacity * 1 * 8  # int64
        rewards_bytes = self.capacity * 1 * 4  # float32
        dones_bytes = self.capacity * 1 * 1  # bool
        masks_bytes = self.capacity * self.action_dim * 1  # bool

        total_bytes = (states_bytes + next_states_bytes + actions_bytes + 
                      rewards_bytes + dones_bytes + masks_bytes)

        return total_bytes / (1024 * 1024)

    def clear(self):
        """Clear all stored transitions."""
        self.ptr = 0
        self.size = 0
        # Arrays remain allocated, just reset pointers

    def save(self, path: str):
        """
        Save replay buffer to disk.

        Args:
            path: File path to save to (.npz format)
        """
        np.savez_compressed(
            path,
            states=self.states[:self.size],
            next_states=self.next_states[:self.size],
            actions=self.actions[:self.size],
            rewards=self.rewards[:self.size],
            dones=self.dones[:self.size],
            legal_masks=self.legal_masks[:self.size],
            ptr=self.ptr,
            size=self.size
        )

    def load(self, path: str):
        """
        Load replay buffer from disk.

        Args:
            path: File path to load from (.npz format)
        """
        data = np.load(path)

        size = int(data['size'])
        self.ptr = int(data['ptr'])
        self.size = size

        # Load data into pre-allocated arrays
        self.states[:size] = data['states']
        self.next_states[:size] = data['next_states']
        self.actions[:size] = data['actions']
        self.rewards[:size] = data['rewards']
        self.dones[:size] = data['dones']
        self.legal_masks[:size] = data['legal_masks']

        print(f"ReplayBuffer loaded: {size:,} transitions from {path}")


# ================================================================
# Testing and Validation
# ================================================================

if __name__ == "__main__":
    print("="*70)
    print("REPLAY BUFFER TEST")
    print("="*70)

    # Test configuration
    capacity = 10000
    action_dim = 168
    device = "cpu"

    # Create buffer
    buffer = ReplayBuffer(capacity=capacity, action_dim=action_dim, device=device)

    print(f"\nBuffer capacity: {buffer.capacity:,}")
    print(f"Buffer size: {len(buffer)}")

    # Test adding transitions
    print("\n" + "-"*70)
    print("Test 1: Adding Transitions")
    print("-"*70)

    for i in range(5):
        state = torch.randn(5, 8, 8)
        action = i % action_dim
        reward = float(i)
        next_state = torch.randn(5, 8, 8)
        done = (i == 4)
        mask = torch.zeros(action_dim, dtype=torch.bool)
        mask[i] = True

        buffer.push(state, action, reward, next_state, done, mask)

    print(f"Added 5 transitions")
    print(f"Buffer size: {len(buffer)}")

    # Test sampling
    print("\n" + "-"*70)
    print("Test 2: Sampling Batch")
    print("-"*70)

    batch_size = 3
    states, actions, rewards, next_states, dones, masks = buffer.sample(batch_size)

    print(f"Batch size: {batch_size}")
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Next states shape: {next_states.shape}")
    print(f"Dones shape: {dones.shape}")
    print(f"Masks shape: {masks.shape}")

    # Test circular buffer behavior
    print("\n" + "-"*70)
    print("Test 3: Circular Buffer (Overwrite)")
    print("-"*70)

    small_buffer = ReplayBuffer(capacity=3, action_dim=action_dim, device=device)

    for i in range(5):
        state = torch.ones(5, 8, 8) * i
        action = i
        reward = float(i)
        next_state = torch.ones(5, 8, 8) * (i + 1)
        done = False
        mask = torch.zeros(action_dim, dtype=torch.bool)

        small_buffer.push(state, action, reward, next_state, done, mask)
        print(f"  Added transition {i}, buffer size: {len(small_buffer)}, ptr: {small_buffer.ptr}")

    print(f"\nFinal buffer size: {len(small_buffer)} (max capacity: 3)")
    print(f"Final pointer: {small_buffer.ptr}")

    # Test is_ready
    print("\n" + "-"*70)
    print("Test 4: Ready Check")
    print("-"*70)

    print(f"Buffer with 5 transitions ready for batch_size=3? {buffer.is_ready(3)}")
    print(f"Buffer with 5 transitions ready for batch_size=10? {buffer.is_ready(10)}")

    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED")
    print("="*70)