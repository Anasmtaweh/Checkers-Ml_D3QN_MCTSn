import torch
import torch.nn as nn


class DuelingDDQNCNN(nn.Module):
    """
    CNN network for Dueling DDQN.
    Input: (12, 8, 8)
    Output: Q-values of shape (batch_size, action_dim)
    """

    def __init__(self, action_dim: int, use_layernorm: bool = False):
        super().__init__()

        ln2 = nn.LayerNorm([64, 8, 8]) if use_layernorm else nn.Identity()
        ln3 = nn.LayerNorm([128, 8, 8]) if use_layernorm else nn.Identity()

        self.conv_body = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            ln2,
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            ln3,
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        ln_value = nn.LayerNorm(256) if use_layernorm else nn.Identity()
        ln_adv = nn.LayerNorm(256) if use_layernorm else nn.Identity()

        self.value_head = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            ln_value,
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.adv_head = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            ln_adv,
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_body(x)
        flat = self.flatten(features)

        value = self.value_head(flat)         # (B, 1)
        advantage = self.adv_head(flat)       # (B, action_dim)

        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        return q_values
