import torch

from training.d3qn.model import D3QNModel
from training.d3qn.trainer import D3QNTrainer
from training.common.board_encoder import CheckersBoardEncoder
from training.common.action_manager import ActionManager
from training.common.replay_buffer import ReplayBuffer


def build_ddqn_trainer(
    env,
    device="cpu",
    gamma=0.99,
    batch_size=64,
    lr=1e-4,
    target_update_interval=1000,
    max_steps_per_episode=200,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=200_000,
    replay_warmup_size=5000,
    use_soft_update=False,
    soft_update_tau=0.0,
    lr_schedule="none",
    lr_gamma=0.99,
    q_clip=0.0,
):
    # Convert string → torch.device without shadowing "device"
    torch_device = torch.device(device)

    encoder = CheckersBoardEncoder()
    action_manager = ActionManager(device=torch_device)
    model = D3QNModel(action_dim=action_manager.action_dim, device=torch_device)

    replay_buffer = ReplayBuffer(
        capacity=50_000,
        device=torch_device,
        action_dim=action_manager.action_dim,
    )

    trainer = D3QNTrainer(
        env=env,
        model=model,
        encoder=encoder,
        replay_buffer=replay_buffer,
        action_manager=action_manager,
        device=torch_device,
        gamma=gamma,
        batch_size=batch_size,
        lr=lr,
        target_update_interval=target_update_interval,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        replay_warmup_size=replay_warmup_size,
        max_steps_per_episode=max_steps_per_episode,
        use_soft_update=use_soft_update,
        soft_update_tau=soft_update_tau,
        lr_schedule=lr_schedule,
        lr_gamma=lr_gamma,
        q_clip=q_clip,
    )

    return trainer
