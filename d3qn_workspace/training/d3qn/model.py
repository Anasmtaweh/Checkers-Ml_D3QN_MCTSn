import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class DuelingDQN(nn.Module):
    def __init__(self, action_dim: int, device: Union[str, torch.device] = "cpu"):
        super(DuelingDQN, self).__init__()

        self.action_dim = action_dim
        self.device = torch.device(device) if isinstance(device, str) else device

        # FIX: Changed in_channels from 5 to 6
        # Channel 0-3: Pieces (My Men, My Kings, Enemy Men, Enemy Kings)
        # Channel 4: Tempo
        # Channel 5: Forced Move Mask (Context)
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.flatten_size = 64 * 8 * 8
        self.fc_norm = nn.LayerNorm(self.flatten_size)

        # Shared fully connected layer
        self.shared_fc = nn.Linear(self.flatten_size, 512) 

        # P1 HEAD (Red)
        self.p1_value = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.p1_advantage = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        # P2 HEAD (Black)
        self.p2_value = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.p2_advantage = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        self._init_weights()
        self.to(self.device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, player_side: int = 1) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        x = x.to(self.device)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = self.fc_norm(x)
        features = F.relu(self.shared_fc(x))

        if player_side == 1:
            val = self.p1_value(features)
            adv = self.p1_advantage(features)
        else:
            val = self.p2_value(features)
            adv = self.p2_advantage(features)

        # Dueling Architecture Aggregation
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def get_q_values(self, state: torch.Tensor, player_side: int = 1) -> torch.Tensor:
        return self.forward(state, player_side)

class D3QNModel:
    def __init__(self, action_dim: int, device: Union[str, torch.device] = "cpu"):
        self.action_dim = action_dim
        self.device = device
        self.online = DuelingDQN(action_dim, device)
        self.target = DuelingDQN(action_dim, device)
        self.update_target_network()
        self.target.eval()

    def update_target_network(self):
        self.target.load_state_dict(self.online.state_dict())

    def get_q_values(self, state, player_side=1, use_target=False):
        net = self.target if use_target else self.online
        return net.get_q_values(state, player_side)
    
    def train(self):
        self.online.train()
    
    def eval(self):
        self.online.eval()
        self.target.eval()
        
    def to(self, device):
        self.device = device
        self.online.to(device)
        self.target.to(device)
        return self