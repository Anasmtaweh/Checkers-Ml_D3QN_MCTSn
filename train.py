"""
Root training script for the D3QN agent in CHECKERS-ML.

Builds a D3QNTrainer, runs training episodes, then evaluation episodes.
"""

import argparse
import os
import torch

from training.d3qn.train_d3qn import build_d3qn_trainer  # pyright: ignore[reportMissingImports]
from training.d3qn.metrics.metric_writer import D3QNMetricWriter
from training.d3qn.metrics.plot_metrics import (
    plot_rewards,
    plot_losses,
    plot_epsilon,
    plot_winrate,
)


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


def apply_preset(preset_name: str, args_obj, defaults):
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


def run_periodic_evaluation(trainer, checkpoint_path, episode, writer, num_games=10, max_moves=300):
    """
    Evaluates the current checkpoint against a Random Agent.
    Uses alternating starting players.
    """
    from training.d3qn.evaluation import evaluate_d3qn_vs_random  # local import to avoid circulars

    stats = evaluate_d3qn_vs_random(
        checkpoint_path,
        num_episodes=num_games,
        device=str(trainer.device),
        max_moves=max_moves,
        verbose=False,
    )

    overall = stats["overall_win_rate"]
    first = stats["first_player_win_rate"]
    second = stats["second_player_win_rate"]
    draws = stats["draw_rate"]
    wins = stats.get("d3qn_total_wins", 0)
    losses = stats.get("random_wins", 0)

    print(
        f"[Eval @ Episode {episode}] "
        f"overall={overall:.2f} first={first:.2f} second={second:.2f} "
        f"(W:{wins}  D:{draws * num_games:.0f}  L:{losses}) "
        f"lr={trainer.optimizer.param_groups[0]['lr']:.6f}"
    )

    if writer:
        # make sure DDQNMetricWriter.write_eval matches this signature
        writer.write_eval(
            episode=episode,
            win_rate=overall,
            first_win_rate=first,
            second_win_rate=second,
            draws=draws,
            lr=trainer.optimizer.param_groups[0]["lr"],
        )

    return stats


def main():
    args = parse_args()
    defaults = vars(args._defaults)

    if args.preset:
        apply_preset(args.preset, args, defaults)

    print("=== CHECKERS-ML: DDQN Training Scaffold ===")
    print(f"Using device: {args.device}")
    print(f"Preset: {args.preset}")

    # Ensure directories exist
    os.makedirs("models/ddqn", exist_ok=True)
    os.makedirs("logs/ddqn", exist_ok=True)
    os.makedirs("logs/ddqn/plots", exist_ok=True)
    os.makedirs("logs/ddqn/metrics", exist_ok=True)

    # Late import to avoid Pylance unbound + circular issues
    from checkers_env.env import CheckersEnv  # pyright: ignore[reportMissingImports]

    # Construct environment
    env = CheckersEnv()

    # Build trainer
    use_soft = args.soft_update or args.use_soft_update
    tau_val = args.soft_tau if args.soft_update_tau is None else args.soft_update_tau
    trainer = build_d3qn_trainer(
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
    writer = D3QNMetricWriter(base_dir="logs/d3qn", run_metadata=run_metadata)
    trainer.metric_writer = writer


    best_score = -float("inf")
    rewards_window = []

    for ep in range(1, args.train_episodes + 1):
        stats = trainer.run_episode(training=True, max_moves=args.max_moves)
        print(
            f"[Train Episode {ep}] reward={stats['total_reward']:.2f} "
            f"moves={stats['moves']} loss={stats.get('loss')}"
        )

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
            # Single, correct eval print (no second conflicting block)
            run_periodic_evaluation(trainer, temp_ckpt, ep, writer, num_games=10, max_moves=args.max_moves)

        if avg_reward > best_score:
            best_score = avg_reward
            trainer.model.save(os.path.join("models/ddqn", "best_model.pt"))

    # Save final model
    save_path = os.path.join("models/ddqn", "final.pt")
    trainer.model.save(save_path)

    # ------------------------------------------------------
    # FINAL EVALUATION AFTER TRAINING
    # ------------------------------------------------------
    print("\n--- Final Evaluation ---")

    from training.d3qn.evaluation import play_game  # pyright: ignore[reportMissingImports]
    from checkers_agents.d3qn_agent import D3QNAgent  # pyright: ignore[reportMissingImports]
    from checkers_agents.random_agent import CheckersRandomAgent  # pyright: ignore[reportMissingImports]

    env = CheckersEnv()
    agent = D3QNAgent(device=args.device)
    agent.load_weights(save_path)
    rand_agent = CheckersRandomAgent()

    # Run 6 evaluation games with alternating starts
    for i in range(6):
        starting_player = 1 if i % 2 == 0 else -1
        env.reset()

        result = play_game(
            env=env,
            agent_first=(agent if starting_player == 1 else rand_agent),
            agent_second=(rand_agent if starting_player == 1 else agent),
            max_moves=400,  # allow longer games
            starting_player=starting_player,
            verbose=False,
        )

        # Normalize reward and winner to be from the D3QN agent's perspective
        d3qn_is_first = starting_player == 1
        d3qn_player_id = result["first_player"] if d3qn_is_first else result["second_player"]

        final_reward = result["total_reward"] if d3qn_is_first else -result["total_reward"]
        
        if result["winner"] == d3qn_player_id:
            final_winner = "D3QN"
        elif result["winner"] == 0 or result["winner"] is None:
            final_winner = "Draw"
        else:
            final_winner = "Random"

        print(
            f"[Final Eval {i+1}] start={'P1' if d3qn_is_first else 'P2'} "
            f"moves={result['moves']} reward={final_reward:.2f} "
            f"winner={final_winner}"
        )

    # Generate plots
    plot_rewards("logs/ddqn/episode_stats.csv", "logs/ddqn/plots/reward_curve.png")
    plot_losses("logs/ddqn/loss.csv", "logs/ddqn/plots/loss_curve.png")
    plot_epsilon("logs/ddqn/episode_stats.csv", "logs/ddqn/plots/epsilon_curve.png")
    plot_winrate("logs/ddqn/metrics/winrate.csv", "logs/ddqn/plots/winrate_curve.png")


if __name__ == "__main__":
    main()
