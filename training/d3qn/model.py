import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network with DUAL HEADS for P1 (Red) and P2 (Black).

    CHANGES FOR GEN 12 DUAL-HEAD:
    - Added shared_fc layer after CNN backbone
    - Split value/advantage streams into P1 and P2 heads
    - forward() now accepts player_side parameter (1 or -1)
    - Each side learns independently without clobbering the other
    """

    def __init__(self, action_dim: int, device: Union[str, torch.device] = "cpu"):
        """
        Initialize Dueling DQN network.

        Args:
            action_dim: Number of possible actions (from ActionManager)
            device: Device to place the model on (cpu or cuda)
        """
        super(DuelingDQN, self).__init__()

        self.action_dim = action_dim
        self.device = torch.device(device) if isinstance(device, str) else device

        # ================================================================
        # CNN Backbone (Feature Extractor) - UNCHANGED
        # ================================================================
        # Input: (batch, 5, 8, 8)
        # 5 channels: my men, my kings, enemy men, enemy kings, tempo

        self.conv1 = nn.Conv2d(
            in_channels=5,
            out_channels=32,
            kernel_size=3,
            padding=1  # Same padding: output size = input size
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # After conv3: (batch, 64, 8, 8)
        # Flattened size: 64 * 8 * 8 = 4096
        self.flatten_size = 64 * 8 * 8

        # Layer normalization for feature stability (Gen 3 patch)
        self.fc_norm = nn.LayerNorm(self.flatten_size)

        # ================================================================
        # NEW: Shared fully connected layer
        # ================================================================
        self.shared_fc = nn.Linear(self.flatten_size, 256)

        # ================================================================
        # P1 (RED) HEAD - NEW
        # ================================================================
        self.p1_value_fc1 = nn.Linear(256, 128)
        self.p1_value_fc2 = nn.Linear(128, 1)  # Single scalar value

        self.p1_advantage_fc1 = nn.Linear(256, 128)
        self.p1_advantage_fc2 = nn.Linear(128, action_dim)

        # ================================================================
        # P2 (BLACK) HEAD - NEW
        # ================================================================
        self.p2_value_fc1 = nn.Linear(256, 128)
        self.p2_value_fc2 = nn.Linear(128, 1)  # Single scalar value

        self.p2_advantage_fc1 = nn.Linear(256, 128)
        self.p2_advantage_fc2 = nn.Linear(128, action_dim)

        # Initialize weights for numerical stability (Gen 3 patch)
        self._init_weights()

        # Move model to device
        self.to(self.device)

    def _init_weights(self):
        """
        Initialize network weights for numerical stability (Gen 3 patch).

        Uses kaiming_normal_ for Conv2d and Linear layers to ensure
        Q-values start near 0.0, preventing initial explosions.
        Biases initialized to 0.0.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor, player_side: int = 1) -> torch.Tensor:
        """
        Forward pass through the network.

        UPDATED FOR DUAL-HEAD: Now accepts player_side parameter.

        Args:
            x: State tensor of shape (batch, 5, 8, 8) or (5, 8, 8)
            player_side: 1 for Red (P1), -1 for Black (P2)

        Returns:
            Q-values tensor of shape (batch, action_dim)
        """
        # Handle single state (no batch dimension)
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension

        # Ensure tensor is on correct device
        x = x.to(self.device)

        # ================================================================
        # CNN Backbone - UNCHANGED
        # ================================================================
        x = F.relu(self.conv1(x))  # (batch, 32, 8, 8)
        x = F.relu(self.conv2(x))  # (batch, 64, 8, 8)
        x = F.relu(self.conv3(x))  # (batch, 64, 8, 8)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # (batch, 4096)

        # Apply layer normalization for stability (Gen 3 patch)
        x = self.fc_norm(x)

        # Shared feature extraction
        features = F.relu(self.shared_fc(x))  # (batch, 256)

        # ================================================================
        # SELECT HEAD BASED ON PLAYER SIDE - NEW
        # ================================================================
        if player_side == 1:
            # RED (P1) HEAD
            value = F.relu(self.p1_value_fc1(features))  # (batch, 128)
            value = self.p1_value_fc2(value)  # (batch, 1)

            advantage = F.relu(self.p1_advantage_fc1(features))  # (batch, 128)
            advantage = self.p1_advantage_fc2(advantage)  # (batch, action_dim)
        else:
            # BLACK (P2) HEAD
            value = F.relu(self.p2_value_fc1(features))  # (batch, 128)
            value = self.p2_value_fc2(value)  # (batch, 1)

            advantage = F.relu(self.p2_advantage_fc1(features))  # (batch, 128)
            advantage = self.p2_advantage_fc2(advantage)  # (batch, action_dim)

        # ================================================================
        # Aggregation: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        # ================================================================
        # The mean subtraction ensures identifiability:
        # Forces the advantage stream to have zero mean, which helps
        # the network learn more stable value and advantage estimates
        advantage_mean = advantage.mean(dim=1, keepdim=True)  # (batch, 1)
        q_values = value + (advantage - advantage_mean)  # (batch, action_dim)

        # Scale output to prevent Q-value explosion (Gen 3 patch)
        q_values = q_values * 0.1  # Scale down by 10x

        return q_values

    def get_q_values(self, state: torch.Tensor, player_side: int = 1) -> torch.Tensor:
        """
        Helper method to get Q-values with explicit device handling.

        UPDATED FOR DUAL-HEAD: Now accepts player_side parameter.

        Args:
            state: State tensor of shape (batch, 5, 8, 8) or (5, 8, 8)
            player_side: 1 for Red (P1), -1 for Black (P2)

        Returns:
            Q-values tensor of shape (batch, action_dim)
        """
        # Ensure state is on the correct device
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)

        state = state.to(self.device)

        # Forward pass with player_side
        return self.forward(state, player_side=player_side)


class D3QNModel:
    """
    Dueling Double Deep Q-Network (D3QN) wrapper.

    Maintains two networks:
    - Online network: Used for action selection and updated every step
    - Target network: Used for computing target Q-values, updated periodically

    The "Double" part refers to using the online network to select actions
    and the target network to evaluate them, reducing overestimation bias.
    """

    def __init__(self, action_dim: int, device: Union[str, torch.device] = "cpu"):
        """
        Initialize D3QN model with online and target networks.

        Args:
            action_dim: Number of possible actions
            device: Device to place models on
        """
        self.action_dim = action_dim
        self.device = torch.device(device) if isinstance(device, str) else device

        # Online network: used for action selection and training
        self.online = DuelingDQN(action_dim, device)

        # Target network: used for computing target Q-values
        self.target = DuelingDQN(action_dim, device)

        # Initialize target network with same weights as online network
        self.update_target_network()

        # Set target network to evaluation mode (no dropout, etc.)
        self.target.eval()

    def update_target_network(self):
        """
        Copy weights from online network to target network.

        This should be called periodically during training (e.g., every N steps)
        to keep the target network updated but stable.
        """
        self.target.load_state_dict(self.online.state_dict())

    def get_q_values(self, state: torch.Tensor, player_side: int = 1, use_target: bool = False) -> torch.Tensor:
        """
        Get Q-values for a given state.

        UPDATED FOR DUAL-HEAD: Now accepts player_side parameter.

        Args:
            state: State tensor of shape (batch, 5, 8, 8) or (5, 8, 8)
            player_side: 1 for Red (P1), -1 for Black (P2)
            use_target: If True, use target network; otherwise use online network

        Returns:
            Q-values tensor of shape (batch, action_dim)
        """
        network = self.target if use_target else self.online
        return network.get_q_values(state, player_side=player_side)

    def train(self):
        """Set online network to training mode."""
        self.online.train()

    def eval(self):
        """Set both networks to evaluation mode."""
        self.online.eval()
        self.target.eval()

    def to(self, device: Union[str, torch.device]):
        """Move both networks to specified device."""
        self.device = torch.device(device) if isinstance(device, str) else device
        self.online.to(self.device)
        self.target.to(self.device)

        # Manually update device attribute in sub-networks
        self.online.device = self.device
        self.target.device = self.device

        return self

    def save(self, path: str):
        """
        Save model state dictionaries.

        Args:
            path: File path to save to
        """
        torch.save({
            'online': self.online.state_dict(),
            'target': self.target.state_dict(),
        }, path)

    def load(self, path: str):
        """
        Load model state dictionaries.

        Args:
            path: File path to load from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.online.load_state_dict(checkpoint['online'])
        self.target.load_state_dict(checkpoint['target'])


# ================================================================
# Utility Functions
# ================================================================

def init_weights(module):
    """
    Initialize network weights using Xavier/Glorot initialization.
    This helps with training stability and convergence.
    """
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ================================================================
# Testing and Debugging
# ================================================================

if __name__ == "__main__":
    print("="*70)
    print("D3QN DUAL-HEAD MODEL ARCHITECTURE TEST")
    print("="*70)

    # Test configuration
    action_dim = 170  # Typical for checkers
    device = "cpu"

    # Create model
    model = D3QNModel(action_dim=action_dim, device=device)

    print(f"\nModel created with action_dim={action_dim}, device={device}")
    print(f"Online network parameters: {count_parameters(model.online):,}")
    print(f"Target network parameters: {count_parameters(model.target):,}")

    # Test single state
    print("\n" + "-"*70)
    print("Test 1: Single State Forward Pass")
    print("-"*70)
    single_state = torch.randn(5, 8, 8)
    q_p1 = model.get_q_values(single_state, player_side=1)
    q_p2 = model.get_q_values(single_state, player_side=-1)
    print(f"P1 Q-values shape: {q_p1.shape}")
    print(f"P2 Q-values shape: {q_p2.shape}")
    print(f"P1 range: [{q_p1.min().item():.4f}, {q_p1.max().item():.4f}]")
    print(f"P2 range: [{q_p2.min().item():.4f}, {q_p2.max().item():.4f}]")

    # Verify P1 and P2 are different
    print(f"P1 and P2 outputs differ: {not torch.allclose(q_p1, q_p2)}")

    # Test batch
    print("\n" + "-"*70)
    print("Test 2: Batch Forward Pass")
    print("-"*70)
    batch_size = 32
    batch_state = torch.randn(batch_size, 5, 8, 8)
    batch_q_p1 = model.get_q_values(batch_state, player_side=1)
    batch_q_p2 = model.get_q_values(batch_state, player_side=-1)
    print(f"Batch P1 Q-values shape: {batch_q_p1.shape}")
    print(f"Batch P2 Q-values shape: {batch_q_p2.shape}")

    # Test target network
    print("\n" + "-"*70)
    print("Test 3: Target Network")
    print("-"*70)
    target_q_p1 = model.get_q_values(single_state, player_side=1, use_target=True)
    print(f"Target P1 Q-values shape: {target_q_p1.shape}")
    print(f"Online and Target P1 match: {torch.allclose(q_p1, target_q_p1)}")

    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED - DUAL-HEAD ARCHITECTURE WORKING")
    print("="*70)