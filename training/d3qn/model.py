import random
from typing import List, Union

import torch
import torch.nn as nn

from training.d3qn.network import DuelingDDQNCNN


class D3QNModel(nn.Module):
    def __init__(self, action_dim: int, device: Union[str, torch.device], network_cls=DuelingDDQNCNN):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.action_dim = action_dim

        self.online = network_cls(action_dim).to(self.device)
        self.target = network_cls(action_dim).to(self.device)

    def get_q_values(self, state_tensor: torch.Tensor) -> torch.Tensor:
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)
        state_tensor = state_tensor.to(self.device)
        return self.online(state_tensor)

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.get_q_values(state_tensor)

    def select_action(self, state_tensor: torch.Tensor, legal_actions: List[int], epsilon: float) -> int:
        if epsilon > 0 and random.random() < epsilon:
            return random.choice(legal_actions)

        q_values = self.get_q_values(state_tensor)  # (B, action_dim)
        q_values = q_values.clone()

        mask = torch.full_like(q_values, -1e9)
        mask[:, legal_actions] = 0
        q_values = q_values + mask

        action_idx = int(torch.argmax(q_values, dim=1).item())
        return action_idx

    def update_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def soft_update_target(self, tau: float) -> None:
        for target_param, online_param in zip(self.target.parameters(), self.online.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def save(self, path: str) -> None:
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
