"""
Utilities for evaluating a DDQN checkpoint across multiple random seeds.
"""

import argparse
import random
from typing import Any, Dict, List

import numpy as np
import torch

from training.ddqn.evaluation import evaluate_ddqn_vs_random


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_checkpoint_multi_seed(
    checkpoint_path: str,
    num_seeds: int,
    episodes_per_seed: int,
    device: str = "cpu",
    max_moves: int = 300,
) -> Dict[str, Any]:
    """
    Evaluate a DDQN checkpoint over multiple random seeds and aggregate statistics.
    """
    win_rates: List[float] = []
    avg_moves: List[float] = []
    avg_rewards: List[float] = []
    per_seed: List[Dict[str, Any]] = []

    for i in range(num_seeds):
        seed_val = i
        _set_global_seed(seed_val)
        stats = evaluate_ddqn_vs_random(
            checkpoint_path=checkpoint_path,
            num_episodes=episodes_per_seed,
            device=device,
            max_moves=max_moves,
            verbose=False,
        )
        win_rates.append(stats.get("ddqn_win_rate", 0.0))
        avg_moves.append(stats.get("avg_moves", 0.0))
        avg_rewards.append(stats.get("avg_ddqn_reward", 0.0))
        per_seed.append({"seed": seed_val, **stats})

    aggregate = {
        "checkpoint": checkpoint_path,
        "num_seeds": num_seeds,
        "episodes_per_seed": episodes_per_seed,
        "win_rate_mean": float(np.mean(win_rates)) if win_rates else 0.0,
        "win_rate_std": float(np.std(win_rates)) if win_rates else 0.0,
        "avg_moves_mean": float(np.mean(avg_moves)) if avg_moves else 0.0,
        "avg_moves_std": float(np.std(avg_moves)) if avg_moves else 0.0,
        "avg_reward_mean": float(np.mean(avg_rewards)) if avg_rewards else 0.0,
        "avg_reward_std": float(np.std(avg_rewards)) if avg_rewards else 0.0,
        "seeds": per_seed,
    }
    return aggregate


def _main():
    parser = argparse.ArgumentParser(description="Multi-seed evaluation for DDQN checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to DDQN checkpoint")
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds to evaluate")
    parser.add_argument("--episodes-per-seed", type=int, default=20, help="Episodes per seed")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--max-moves", type=int, default=300, help="Max moves per episode")
    args = parser.parse_args()

    aggregate = evaluate_checkpoint_multi_seed(
        checkpoint_path=args.checkpoint,
        num_seeds=args.num_seeds,
        episodes_per_seed=args.episodes_per_seed,
        device=args.device,
        max_moves=args.max_moves,
    )
    print("=== Multi-seed Evaluation ===")
    print(f"Checkpoint: {aggregate['checkpoint']}")
    print(f"Seeds: {aggregate['num_seeds']}, Episodes/seed: {aggregate['episodes_per_seed']}")
    print(
        f"Win-rate mean={aggregate['win_rate_mean']:.3f} std={aggregate['win_rate_std']:.3f}, "
        f"Avg moves mean={aggregate['avg_moves_mean']:.1f} std={aggregate['avg_moves_std']:.1f}, "
        f"Avg reward mean={aggregate['avg_reward_mean']:.3f} std={aggregate['avg_reward_std']:.3f}"
    )


if __name__ == "__main__":
    _main()
