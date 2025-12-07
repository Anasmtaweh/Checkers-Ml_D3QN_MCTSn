from typing import Dict, Union

import torch


class ReplayBuffer:
    def __init__(self, capacity: int, device: Union[str, torch.device], action_dim: int = 4032):
        self.capacity = capacity
        self.device = torch.device(device) if isinstance(device, str) else device
        self.action_dim = action_dim

        self.states = torch.empty((capacity, 12, 8, 8), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((capacity,), dtype=torch.long, device=self.device)
        self.rewards = torch.empty((capacity,), dtype=torch.float32, device=self.device)
        self.next_states = torch.empty((capacity, 12, 8, 8), dtype=torch.float32, device=self.device)
        self.dones = torch.empty((capacity,), dtype=torch.bool, device=self.device)
        self.legal_masks_next = [None] * capacity
        self.legal_masks_current = [None] * capacity

        self.size = 0
        self.ptr = 0

    def add(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool, legal_mask_next=None, legal_mask_current=None) -> None:
        state = state.squeeze(0)
        next_state = next_state.squeeze(0)

        idx = self.ptr
        self.states[idx].copy_(state)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.next_states[idx].copy_(next_state)
        self.dones[idx] = bool(done)
        if legal_mask_next is not None:
            self.legal_masks_next[idx] = (legal_mask_next.to(self.device) > 0).float()
        else:
            self.legal_masks_next[idx] = None
        if legal_mask_current is not None:
            self.legal_masks_current[idx] = (legal_mask_current.to(self.device) > 0).float()
        else:
            self.legal_masks_current[idx] = None

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if self.size < batch_size:
            raise ValueError("Not enough samples in replay buffer to sample the requested batch size.")

        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        batch = {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
        }

        # Next-state masks
        masks_next = []
        any_mask_next = False
        for idx in indices.tolist():
            m = self.legal_masks_next[idx]
            if m is not None:
                any_mask_next = True
            masks_next.append(m)

        if any_mask_next:
            fixed_masks_next = []
            for m in masks_next:
                if m is None:
                    fixed_masks_next.append(torch.ones(self.action_dim, device=self.device, dtype=torch.float32))
                else:
                    mask_t = (m.to(self.device) > 0).float()
                    fixed_masks_next.append(mask_t if mask_t.shape[0] == self.action_dim else torch.ones(self.action_dim, device=self.device, dtype=torch.float32))
            batch["legal_masks_next"] = torch.stack(fixed_masks_next, dim=0)

        # Current-state masks
        masks_current = []
        any_mask_current = False
        for idx in indices.tolist():
            m = self.legal_masks_current[idx]
            if m is not None:
                any_mask_current = True
            masks_current.append(m)

        if any_mask_current:
            fixed_masks_current = []
            for m in masks_current:
                if m is None:
                    fixed_masks_current.append(torch.ones(self.action_dim, device=self.device, dtype=torch.float32))
                else:
                    mask_t = (m.to(self.device) > 0).float()
                    fixed_masks_current.append(mask_t if mask_t.shape[0] == self.action_dim else torch.ones(self.action_dim, device=self.device, dtype=torch.float32))
            batch["legal_masks_current"] = torch.stack(fixed_masks_current, dim=0)

        return batch

    def __len__(self) -> int:
        return self.size
