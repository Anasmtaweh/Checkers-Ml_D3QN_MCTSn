"""
Round-robin evaluation across multiple DDQN checkpoints.

Usage:
  python -m training.ddqn.tournament_eval --checkpoints models/ddqn/best_model.pt,models/ddqn/final.pt --episodes 20
"""

import argparse
import glob
import os
from typing import Dict, List, Tuple

from training.ddqn import evaluate_ddqn_vs_ddqn


def _gather_checkpoints(pattern: str) -> List[str]:
    if "," in pattern:
        return [p.strip() for p in pattern.split(",") if p.strip()]
    if os.path.isdir(pattern):
        return sorted(glob.glob(os.path.join(pattern, "*.pt")))
    return sorted(glob.glob(pattern))


def run_round_robin(checkpoints: List[str], episodes: int, device: str, max_moves: int) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    scores: Dict[str, float] = {ckpt: 0.0 for ckpt in checkpoints}
    results: Dict[str, Dict[str, float]] = {ckpt: {} for ckpt in checkpoints}

    for i, ckpt_a in enumerate(checkpoints):
        for ckpt_b in checkpoints[i + 1 :]:
            stats = evaluate_ddqn_vs_ddqn(
                checkpoint_a=ckpt_a,
                checkpoint_b=ckpt_b,
                num_episodes=episodes,
                device=device,
                max_moves=max_moves,
                verbose=False,
            )
            # simple Elo-like scoring: win=1, draw=0.5
            a_pts = stats["agent_a_wins"] + 0.5 * stats["draws"]
            b_pts = stats["agent_b_wins"] + 0.5 * stats["draws"]
            scores[ckpt_a] += a_pts
            scores[ckpt_b] += b_pts
            results[ckpt_a][ckpt_b] = stats["agent_a_win_rate"]
            results[ckpt_b][ckpt_a] = stats["agent_b_win_rate"]
            print(
                f"{os.path.basename(ckpt_a)} vs {os.path.basename(ckpt_b)} -> "
                f"A win_rate={stats['agent_a_win_rate']:.2f}, B win_rate={stats['agent_b_win_rate']:.2f}, draws={stats['draw_rate']:.2f}"
            )

    return scores, results


def main():
    parser = argparse.ArgumentParser(description="Round-robin tournament for DDQN checkpoints")
    parser.add_argument("--checkpoints", type=str, required=True, help="Comma list, glob, or directory of .pt files")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per matchup")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--max-moves", type=int, default=300, help="Max moves per game")
    args = parser.parse_args()

    ckpts = _gather_checkpoints(args.checkpoints)
    if len(ckpts) < 2:
        raise ValueError("Need at least two checkpoints for a tournament.")

    print(f"Running round-robin on {len(ckpts)} checkpoints: {[os.path.basename(c) for c in ckpts]}")
    scores, results = run_round_robin(ckpts, args.episodes, args.device, args.max_moves)

    print("\n=== Tournament Scores (higher is better) ===")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (ckpt, score) in enumerate(sorted_scores, 1):
        print(f"{rank}. {os.path.basename(ckpt)} -> {score:.2f} pts")


if __name__ == "__main__":
    main()
