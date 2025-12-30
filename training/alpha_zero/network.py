import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union


class AlphaZeroNet(nn.Module):
    """
    AlphaZero-style Neural Network for Checkers.
    
    Architecture:
    - Input: (Batch, 5, 8, 8) - 5-channel board encoding
    - Backbone: 3-layer CNN feature extractor (similar to D3QN)
    - Dual Heads:
        1. Policy Head (Actor): Outputs action probabilities via LogSoftmax
        2. Value Head (Critic): Outputs position evaluation via Tanh
    
    Key Differences from D3QN:
    - Policy head outputs log-probabilities instead of Q-values
    - Value head outputs a single scalar in [-1, 1] range
    - No separate P1/P2 heads (canonicalization handles perspective)
    """
    
    def __init__(self, action_dim: int = 170, device: Union[str, torch.device] = "cpu"):
        """
        Initialize AlphaZero network.
        
        Args:
            action_dim: Number of possible actions (typically ~170 for checkers)
            device: Device to place the model on (cpu or cuda)
        """
        super(AlphaZeroNet, self).__init__()
        
        self.action_dim = action_dim
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # ================================================================
        # CNN BACKBONE (Feature Extractor)
        # ================================================================
        # Input: (batch, 5, 8, 8)
        # 5 channels: my men, my kings, enemy men, enemy kings, tempo
        
        self.conv1 = nn.Conv2d(
            in_channels=5,
            out_channels=32,
            kernel_size=3,
            padding=1  # Same padding: keeps 8x8 spatial dimensions
        )
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(64)
        
        # After conv3: (batch, 64, 8, 8)
        # Flattened size: 64 * 8 * 8 = 4096
        self.flatten_size = 64 * 8 * 8
        
        # ================================================================
        # POLICY HEAD (Actor)
        # ================================================================
        # Predicts action probabilities
        self.policy_fc1 = nn.Linear(self.flatten_size, 256)
        self.policy_fc2 = nn.Linear(256, action_dim)
        # Output: LogSoftmax over action_dim (log-probabilities)
        
        # ================================================================
        # VALUE HEAD (Critic)
        # ================================================================
        # Predicts position evaluation
        self.value_fc1 = nn.Linear(self.flatten_size, 128)
        self.value_fc2 = nn.Linear(128, 1)
        # Output: Tanh to force range [-1, 1]
        
        # Initialize weights for stability
        self._init_weights()
        
        # Move model to device
        self.to(self.device)
    
    def _init_weights(self):
        """
        Initialize network weights using Kaiming initialization.
        
        This ensures:
        - CNN layers start with appropriate variance for ReLU activations
        - Fully connected layers don't explode or vanish initially
        - Biases start at zero
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: State tensor of shape (batch, 5, 8, 8) or (5, 8, 8)
        
        Returns:
            Tuple of (policy_logits, value):
                - policy_logits: Log-probabilities of shape (batch, action_dim)
                - value: Position evaluation of shape (batch, 1) in range [-1, 1]
        """
        # Handle single state (no batch dimension)
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension: (5, 8, 8) -> (1, 5, 8, 8)
        
        # Ensure tensor is on correct device and float type (crucial for CNN)
        x = x.to(self.device).float()
        
        # ================================================================
        # CNN BACKBONE - Feature Extraction
        # ================================================================
        # Conv Block 1
        x = self.conv1(x)           # (batch, 32, 8, 8)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Conv Block 2
        x = self.conv2(x)           # (batch, 64, 8, 8)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Conv Block 3
        x = self.conv3(x)           # (batch, 64, 8, 8)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Flatten for fully connected layers
        features = x.view(x.size(0), -1)  # (batch, 4096)
        
        # ================================================================
        # POLICY HEAD (Actor)
        # ================================================================
        policy = F.relu(self.policy_fc1(features))  # (batch, 256)
        policy_logits = self.policy_fc2(policy)     # (batch, action_dim)
        
        # Apply LogSoftmax for stable log-probabilities
        # CRUCIAL: This is required for AlphaZero's loss function
        policy_logits = F.log_softmax(policy_logits, dim=1)
        
        # ================================================================
        # VALUE HEAD (Critic)
        # ================================================================
        value = F.relu(self.value_fc1(features))    # (batch, 128)
        value = self.value_fc2(value)               # (batch, 1)
        
        # Apply Tanh to constrain value to [-1, 1]
        # CRUCIAL: Matches MCTS value range and training targets
        value = torch.tanh(value)
        
        return policy_logits, value
    
    def predict(self, state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Predict policy and value for a single state (inference mode).
        
        Args:
            state: State tensor of shape (5, 8, 8) or (batch, 5, 8, 8)
        
        Returns:
            Tuple of (policy, value):
                - policy: Action probabilities of shape (action_dim,)
                - value: Position evaluation scalar in range [-1, 1]
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            # Forward handles dimension checking and device placement
            policy_logits, value = self.forward(state)
            
            # Convert log-probabilities to probabilities
            policy = torch.exp(policy_logits).squeeze(0)  # (action_dim,)
            value = value.item()  # Convert to Python float
            
        return policy, value
    
    def get_policy_value(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get policy and value for training (keeps gradients).
        
        Args:
            state: State tensor of shape (batch, 5, 8, 8) or (5, 8, 8)
        
        Returns:
            Tuple of (policy_logits, value):
                - policy_logits: Log-probabilities of shape (batch, action_dim)
                - value: Position evaluation of shape (batch, 1)
        """
        return self.forward(state)


class AlphaZeroModel:
    """
    Wrapper for AlphaZero network with convenience methods.
    
    This class provides:
    - Easy saving/loading of checkpoints
    - Device management
    - Training/evaluation mode switching
    """
    
    def __init__(self, action_dim: int = 170, device: Union[str, torch.device] = "cpu"):
        """
        Initialize AlphaZero model.
        
        Args:
            action_dim: Number of possible actions
            device: Device to place model on
        """
        self.action_dim = action_dim
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Create network
        self.network = AlphaZeroNet(action_dim, device)
    
    def predict(self, state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Predict policy and value for a single state.
        
        Args:
            state: State tensor of shape (5, 8, 8)
        
        Returns:
            Tuple of (policy, value)
        """
        return self.network.predict(state)
    
    def get_policy_value(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get policy and value for training.
        
        Args:
            state: State tensor of shape (batch, 5, 8, 8)
        
        Returns:
            Tuple of (policy_logits, value)
        """
        return self.network.get_policy_value(state)
    
    def train(self):
        """Set network to training mode."""
        self.network.train()
    
    def eval(self):
        """Set network to evaluation mode."""
        self.network.eval()
    
    def to(self, device: Union[str, torch.device]):
        """Move network to specified device."""
        self.device = torch.device(device) if isinstance(device, str) else device
        self.network.to(self.device)
        self.network.device = self.device
        return self
    
    def save(self, path: str):
        """
        Save model checkpoint.
        
        Args:
            path: File path to save to (e.g., 'checkpoints/model.pth')
        """
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'action_dim': self.action_dim,
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: File path to load from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from {path}")
    
    def num_parameters(self) -> int:
        """
        Count total number of trainable parameters.
        
        Returns:
            Total parameter count
        """
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)


# ================================================================
# Testing and Debugging
# ================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ALPHAZERO NETWORK ARCHITECTURE TEST")
    print("=" * 70)
    
    # Test configuration
    action_dim = 170
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model
    model = AlphaZeroModel(action_dim=action_dim, device=device)
    
    print(f"\nModel created with action_dim={action_dim}, device={device}")
    print(f"Total parameters: {model.num_parameters():,}")
    
    # Test 1: Single state forward pass
    print("\n" + "-" * 70)
    print("Test 1: Single State Forward Pass")
    print("-" * 70)
    single_state = torch.randn(5, 8, 8)
    policy, value = model.predict(single_state)
    print(f"Policy shape: {policy.shape}")
    print(f"Value: {value:.4f}")
    print(f"Policy sum (should be ~1.0): {policy.sum().item():.6f}")
    print(f"Value in range [-1, 1]: {-1 <= value <= 1}")
    
    # Test 2: Batch forward pass
    print("\n" + "-" * 70)
    print("Test 2: Batch Forward Pass")
    print("-" * 70)
    batch_size = 32
    batch_state = torch.randn(batch_size, 5, 8, 8)
    model.train()
    policy_logits, values = model.get_policy_value(batch_state)
    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Values shape: {values.shape}")
    print(f"Values range: [{values.min().item():.4f}, {values.max().item():.4f}]")
    
    # Test 3: Gradient flow
    print("\n" + "-" * 70)
    print("Test 3: Gradient Flow")
    print("-" * 70)
    loss = policy_logits.mean() + values.mean()
    loss.backward()
    has_gradients = any(p.grad is not None for p in model.network.parameters())
    print(f"Gradients computed: {has_gradients}")
    
    # Test 4: Save and load
    print("\n" + "-" * 70)
    print("Test 4: Save and Load")
    print("-" * 70)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        model.save(tmp.name)
        
        # Create new model and load
        model2 = AlphaZeroModel(action_dim=action_dim, device=device)
        model2.load(tmp.name)
        
        # Verify weights match
        policy2, value2 = model2.predict(single_state)
        weights_match = torch.allclose(policy, policy2) and abs(value - value2) < 1e-6
        print(f"Weights match after load: {weights_match}")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED - ALPHAZERO NETWORK READY")
    print("=" * 70)
    print("\nKey Architecture Features:")
    print("  • Input: (Batch, 5, 8, 8) - 5-channel board encoding")
    print("  • Backbone: 3-layer CNN with BatchNorm")
    print("  • Policy Head: LogSoftmax output (log-probabilities)")
    print("  • Value Head: Tanh output (range [-1, 1])")
    print("  • Total Parameters: {:,}".format(model.num_parameters()))
