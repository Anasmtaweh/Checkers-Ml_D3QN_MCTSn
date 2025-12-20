import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class D3QNTrainer:
    """
    Dueling Double Deep Q-Network Trainer (Stabilized).
    """

    def __init__(
        self,
        env,
        action_manager,
        board_encoder,
        model,
        optimizer,
        buffer,
        device,
        gamma=0.99,
        gradient_clip=1.0,
        loss_type="huber",
        tau=0.001
    ):
        self.env = env
        self.action_manager = action_manager
        self.board_encoder = board_encoder
        self.model = model
        self.optimizer = optimizer
        self.buffer = buffer
        self.device = device
        self.gamma = gamma
        self.gradient_clip = gradient_clip
        self.loss_type = loss_type
        self.tau = tau

    def train_step(self, batch_size):
        # 1. Sample
        state, action, reward, next_state, done, next_legal_mask = self.buffer.sample(batch_size)

        # 2. Compute Current Q
        q_values = self.model.online(state)
        
        # FIX: 'action' is already [batch, 1], so we don't need unsqueeze.
        # We also keep q_value as [batch, 1] to match expected_q_value shape.
        q_value = q_values.gather(1, action)

        # 3. Compute Target Q
        with torch.no_grad():
            next_q_online = self.model.online(next_state)
            next_q_online[~next_legal_mask] = -float('inf')
            next_action = next_q_online.argmax(1).unsqueeze(1)
            
            next_q_target = self.model.target(next_state)
            next_q_value = next_q_target.gather(1, next_action)
            
            # This calculation stays [batch, 1]
            expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        # 4. Loss
        loss = F.smooth_l1_loss(q_value, expected_q_value)

        # 5. Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.online.parameters(), self.gradient_clip)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        for target_param, online_param in zip(self.model.target.parameters(), self.model.online.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)