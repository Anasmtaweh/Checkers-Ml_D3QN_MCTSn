"""
Root training script for the DDQN agent in CHECKERS-ML.

Builds a DDQNTrainer, runs training episodes, then evaluation episodes.
"""

import argparse
import os
import torch
from training.ddqn.train_ddqn import build_ddqn_trainer

from checkers_env.env import CheckersEnv
from training.ddqn.train_ddqn import build_ddqn_trainer  # pyright: ignore[reportMissingImports]
from training.ddqn.metrics.metric_writer import DDQNMetricWriter
from training.ddqn.metrics.plot_metrics import (
    plot_rewards,
    plot_losses,
    plot_epsilon,
    plot_winrate,
)
from training.ddqn.evaluation import evaluate_ddqn_vs_random


def parse_args():
    parser = argparse.ArgumentParser(description="DDQN Training Script")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--train-episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--eval-episodes", type=int, default=3, help="Number of evaluation episodes after training")
    parser.add_argument("--max-moves", type=int, default=200, help="Max moves per episode")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for DDQN updates")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (e.g., 1e-4 or 5e-4 recommended for stable long runs)",
    )
    parser.add_argument(
        "--target-update",
        type=int,
        default=1000,
        help="Target network HARD update interval (steps); ignored if soft-update is enabled",
    )
    parser.add_argument(
        "--replay-warmup-size",
        type=int,
        default=1000,
        help="Steps before starting training updates",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=500,
        help="Episodes between checkpoints",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="Episodes between evaluation runs (0 to disable)",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        choices=["none", "exponential", "cosine"],
        default="none",
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.99,
        help="Gamma for LR scheduler (if applicable)",
    )
    parser.add_argument(
        "--qclip",
        type=float,
        default=0.0,
        help="Clamp Q-values to [-qclip, qclip] (0 to disable)",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Starting epsilon for exploration (default 1.0)",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="Final epsilon after decay (default 0.05)",
    )
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=100_000,
        help="Global steps to decay epsilon (e.g., 100k-300k recommended for Checkers)",
    )
    parser.add_argument(
        "--soft-update",
        action="store_true",
        help="Use soft target network updates instead of hard",
    )
    parser.add_argument(
        "--soft-tau",
        type=float,
        default=0.0,
        help="Tau for soft updates (e.g., 0.005)",
    )
    parser.add_argument(
        "--use-soft-update",
        action="store_true",
        help="Alias for --soft-update",
    )
    parser.add_argument(
        "--soft-update-tau",
        type=float,
        default=None,
        help="Alias for --soft-tau",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["debug", "baseline", "soft-update", "stable-checkers"],
        default=None,
        help=(
            "Use a preset configuration (debug/baseline/soft-update/stable-checkers) "
            "that can be overridden by explicit flags."
        ),
    )

    # Parse twice: once for defaults snapshot, once for real args
    defaults = parser.parse_args([])
    args = parser.parse_args()
    args._defaults = defaults
    return args


def main():
    args = parse_args()
    defaults = vars(args._defaults)

    # Apply preset overrides where the user did not explicitly change values
    def apply_preset(preset_name: str, args_obj):
        presets = {
            "debug": {
                "train_episodes": 50,
                "max_moves": 80,
                "epsilon_decay_steps": 10_000,
                "replay_warmup_size": 200,
                "batch_size": 32,
                "eval_interval": 5,
                "save_interval": 100,
            },
            "baseline": {
                "train_episodes": 10000,
                "lr": 1e-4,
                "epsilon_decay_steps": 200_000,
                "replay_warmup_size": 2000,
                "save_interval": 500,
                "eval_interval": 25,
            },
            "soft-update": {
                "train_episodes": 10000,
                "lr": 1e-4,
                "epsilon_decay_steps": 200_000,
                "replay_warmup_size": 2000,
                "save_interval": 500,
                "eval_interval": 25,
                "soft_update": True,
                "use_soft_update": True,
                "soft_tau": 0.005,
                "soft_update_tau": 0.005,
            },
            # Tuned preset for stable Checkers DDQN training
            "stable-checkers": {
                # Training length
                "train_episodes": 20000,
                "max_moves": 200,

                # Optimizer / LR
                "lr": 1e-4,

                # Replay / batch
                "batch_size": 64,
                "replay_warmup_size": 5000,

                # Exploration
                "epsilon_start": 1.0,
                "epsilon_end": 0.05,
                "epsilon_decay_steps": 200_000,

                # Target updates (soft)
                "soft_update": True,
                "use_soft_update": True,
                "soft_tau": 0.005,
                "soft_update_tau": 0.005,

                # Logging / eval
                "save_interval": 500,
                "eval_interval": 25,

                # LR schedule & Q clipping
                "lr_schedule": "none",
                "lr_gamma": 0.99,
                "qclip": 10.0,
            },
        }
        if preset_name not in presets:
            return
        preset_values = presets[preset_name]
        for key, val in preset_values.items():
            if hasattr(args_obj, key):
                # Only override if user kept default
                if getattr(args_obj, key) == defaults.get(key):
                    setattr(args_obj, key, val)

    if args.preset:
        apply_preset(args.preset, args)

    print("=== CHECKERS-ML: DDQN Training Scaffold ===")
    print(f"Using device: {args.device}")
    print(f"Preset: {args.preset}")

    # Ensure directories exist
    os.makedirs("models/ddqn", exist_ok=True)
    os.makedirs("logs/ddqn", exist_ok=True)
    os.makedirs("logs/ddqn/plots", exist_ok=True)
    os.makedirs("logs/ddqn/metrics", exist_ok=True)

    # Construct environment
    env = CheckersEnv()

    # Build trainer
    use_soft = args.soft_update or args.use_soft_update
    tau_val = args.soft_tau if args.soft_update_tau is None else args.soft_update_tau
    trainer = build_ddqn_trainer(
        env,
        device=args.device,
        gamma=args.gamma,
        batch_size=args.batch_size,
        lr=args.lr,
        target_update_interval=args.target_update,
        max_steps_per_episode=args.max_moves,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        replay_warmup_size=args.replay_warmup_size,
        use_soft_update=use_soft,
        soft_update_tau=tau_val,
        lr_schedule=args.lr_schedule,
        lr_gamma=args.lr_gamma,
        q_clip=args.qclip,
    )

    print("\nTrainer successfully constructed.")
    print("Model:", trainer.model)
    print("Action dimension:", trainer.action_manager.action_dim)
    print(
        f"Hyperparameters -> lr={args.lr}, epsilon_start={args.epsilon_start}, "
        f"epsilon_end={args.epsilon_end}, decay_steps={args.epsilon_decay_steps}, "
        f"use_soft_update={use_soft}, tau={tau_val}, target_update_interval={args.target_update}, "
        f"replay_warmup_size={args.replay_warmup_size}, preset={args.preset}, "
        f"lr_schedule={args.lr_schedule}, lr_gamma={args.lr_gamma}, q_clip={args.qclip}"
    )

    # Training phase
    print("\n--- Training ---")
    run_metadata = {
        "seed": None,
        "device": args.device,
        "gamma": args.gamma,
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "lr_gamma": args.lr_gamma,
        "batch_size": args.batch_size,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_steps": args.epsilon_decay_steps,
        "soft_update_tau": tau_val,
        "use_soft_update": use_soft,
        "train_episodes": args.train_episodes,
        "eval_episodes": args.eval_episodes,
        "max_moves": args.max_moves,
        "replay_warmup_size": args.replay_warmup_size,
        "preset": args.preset,
        "q_clip": args.qclip,
    }
    writer = DDQNMetricWriter(base_dir="logs/ddqn", run_metadata=run_metadata)
    trainer.metric_writer = writer

    best_score = -float("inf")
    rewards_window = []

    def run_periodic_evaluation(trainer_obj, checkpoint_path, episode, writer_obj):
        stats = evaluate_ddqn_vs_random(
            checkpoint_path=checkpoint_path,
            num_episodes=10,
            device=trainer_obj.device,
            max_moves=trainer_obj.max_steps_per_episode,
        )
        writer_obj.log_winrate(
            episode=episode,
            ddqn_wins=stats["ddqn_wins"],
            random_wins=stats["random_wins"],
            draws=stats["draws"],
            win_rate=stats["ddqn_win_rate"],
        )
        return stats

    for ep in range(1, args.train_episodes + 1):
        stats = trainer.run_episode(training=True, max_moves=args.max_moves)
        rewards_window.append(stats["total_reward"])
        if len(rewards_window) > 20:
            rewards_window.pop(0)
        avg_reward = sum(rewards_window) / len(rewards_window)

        writer.log_episode(
            episode=ep,
            reward=stats["total_reward"],
            epsilon=trainer.epsilon,
            replay_size=len(trainer.replay_buffer),
            moves=stats["moves"],
            loss=stats.get("loss"),
        )

        if ep % args.save_interval == 0:
            ckpt_path = os.path.join("models/ddqn", f"checkpoint_{ep}.pt")
            trainer.model.save(ckpt_path)

        if args.eval_interval and ep % args.eval_interval == 0:
            temp_ckpt = os.path.join("models/ddqn", "temp_eval.pt")
            trainer.model.save(temp_ckpt)
            eval_stats = run_periodic_evaluation(trainer, temp_ckpt, ep, writer)
            current_lr = trainer.optimizer.param_groups[0]["lr"]
            print(
                f"[Eval @ Episode {ep}] win_rate={eval_stats['ddqn_win_rate']:.2f} "
                f"wins={eval_stats['ddqn_wins']} losses={eval_stats['random_wins']} draws={eval_stats['draws']} "
                f"lr={current_lr:.6f}"
            )

        if avg_reward > best_score:
            best_score = avg_reward
            trainer.model.save(os.path.join("models/ddqn", "best_model.pt"))

    # Save final model
    trainer.model.save(os.path.join("models/ddqn", "final.pt"))

    # Evaluation phase
    print("\n--- Evaluation ---")
    for ep in range(args.eval_episodes):
        stats = trainer.run_episode(training=False, max_moves=args.max_moves)
        print(
            f"[Eval {ep+1}] moves={stats['moves']} "
            f"reward={stats['total_reward']} winner={stats['winner']}"
        )

    # Generate plots
    plot_rewards("logs/ddqn/episode_stats.csv", "logs/ddqn/plots/reward_curve.png")
    plot_losses("logs/ddqn/loss.csv", "logs/ddqn/plots/loss_curve.png")
    plot_epsilon("logs/ddqn/episode_stats.csv", "logs/ddqn/plots/epsilon_curve.png")
    plot_winrate("logs/ddqn/metrics/winrate.csv", "logs/ddqn/plots/winrate_curve.png")


if __name__ == "__main__":
    main()

