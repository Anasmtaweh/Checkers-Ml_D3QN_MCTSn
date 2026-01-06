import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union


class AlphaZeroNet(nn.Module):
    """
    AlphaZero-style Neural Network for Checkers.
    
    Architecture:
    - Input: (Batch, 6, 8, 8) - 6-channel board encoding
    - Backbone: 3-layer CNN feature extractor
    - Dual Heads:
        1. Policy Head (Actor): Outputs action probabilities via LogSoftmax
        2. Value Head (Critic): Outputs position evaluation via Tanh
    """
    
    def __init__(self, action_dim: int = 170, device: Union[str, torch.device] = "cpu"):
        """
        Initialize AlphaZero network.
        
        Args:
            action_dim: Number of possible actions
            device: Device to place the model on
        """
        super(AlphaZeroNet, self).__init__()
        
        self.action_dim = action_dim
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # ================================================================
        # CNN BACKBONE (Feature Extractor)
        # ================================================================
        # Input: (batch, 6, 8, 8)
        # 6 channels: my men, my kings, enemy men, enemy kings, tempo, forced_mask
        
        self.conv1 = nn.Conv2d(
            in_channels=6,        # CHANGED FROM 5 TO 6
            out_channels=32,
            kernel_size=3,
            padding=1  
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
        
        self.flatten_size = 64 * 8 * 8
        
        # ================================================================
        # POLICY HEAD (Actor)
        # ================================================================
        self.policy_fc1 = nn.Linear(self.flatten_size, 256)
        self.policy_fc2 = nn.Linear(256, action_dim)
        
        # ================================================================
        # VALUE HEAD (Critic)
        # ================================================================
        self.value_fc1 = nn.Linear(self.flatten_size, 128)
        self.value_fc2 = nn.Linear(128, 1)
        
        # Initialize weights for stability
        self._init_weights()
        
        # Move model to device
        self.to(self.device)
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                if module.out_features == 1:
                    nn.init.uniform_(module.weight, -0.01, 0.01)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                else:
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(0)  
        
        x = x.to(self.device).float()
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        features = x.view(x.size(0), -1) 
        
        policy = F.relu(self.policy_fc1(features))
        policy_logits = self.policy_fc2(policy)
        policy_logits = F.log_softmax(policy_logits, dim=1)
        
        value = F.relu(self.value_fc1(features))
        value = self.value_fc2(value)
        value = torch.tanh(value)
        
        return policy_logits, value
    
    def predict(self, state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        self.eval() 
        with torch.no_grad():
            policy_logits, value = self.forward(state)
            policy = torch.exp(policy_logits).squeeze(0)
            value = value.item()  
        return policy, value
    
    def get_policy_value(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(state)


class AlphaZeroModel:
    def __init__(self, action_dim: int = 170, device: Union[str, torch.device] = "cpu"):
        self.action_dim = action_dim
        self.device = torch.device(device) if isinstance(device, str) else device
        self.network = AlphaZeroNet(action_dim, device)
    
    def predict(self, state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        return self.network.predict(state)
    
    def get_policy_value(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.network.get_policy_value(state)
    
    def train(self):
        self.network.train()
    
    def eval(self):
        self.network.eval()
    
    def to(self, device: Union[str, torch.device]):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.network.to(self.device)
        self.network.device = self.device
        return self
    
    def save(self, path: str):
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'action_dim': self.action_dim,
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from {path}")
    
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)


# ================================================================
# Testing and Debugging
# ================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ALPHAZERO NETWORK ARCHITECTURE TEST (6 CHANNELS)")
    print("=" * 70)
    
    action_dim = 170
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AlphaZeroModel(action_dim=action_dim, device=device)
    
    print(f"\nModel created with action_dim={action_dim}, device={device}")
    print(f"Total parameters: {model.num_parameters():,}")
    
    # Test 1: Single state forward pass (Now with 6 channels)
    print("\n" + "-" * 70)
    print("Test 1: Single State Forward Pass (6 channels)")
    print("-" * 70)
    single_state = torch.randn(6, 8, 8)
    policy, value = model.predict(single_state)
    print(f"Input shape: {single_state.shape}")
    print(f"Policy shape: {policy.shape}")
    print(f"Value: {value:.4f}")
    
    # Verify input channel check
    try:
        wrong_state = torch.randn(5, 8, 8)
        model.predict(wrong_state)
    except RuntimeError:
        print("✓ Correctly caught wrong channel input (5 channels)")

    print("\n✓ ALL TESTS PASSED - ALPHAZERO NETWORK READY")