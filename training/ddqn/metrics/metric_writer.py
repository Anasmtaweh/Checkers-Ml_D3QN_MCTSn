import csv
import json
import os
from typing import Optional


class DDQNMetricWriter:
    def __init__(self, base_dir: str = "logs/ddqn", run_metadata: Optional[dict] = None):
        self.base_dir = base_dir
        self.metrics_dir = os.path.join(base_dir, "metrics")
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)

        self.episode_jsonl_path = os.path.join(self.metrics_dir, "episode_stats.jsonl")
        self.episode_csv_path = os.path.join(self.base_dir, "episode_stats.csv")
        self.loss_csv_path = os.path.join(self.base_dir, "loss.csv")
        self.winrate_csv_path = os.path.join(self.metrics_dir, "winrate.csv")

        # Initialize CSV headers if files are new
        if not os.path.exists(self.episode_csv_path):
            with open(self.episode_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "reward", "epsilon", "replay", "moves", "loss"])

        if not os.path.exists(self.loss_csv_path):
            with open(self.loss_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "loss"])

        if not os.path.exists(self.winrate_csv_path):
            with open(self.winrate_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "ddqn_wins", "random_wins", "draws", "win_rate"])

        # Persist run metadata once per run
        if run_metadata is not None:
            metadata_path = os.path.join(self.base_dir, "run_metadata.json")
            try:
                with open(metadata_path, "w") as mf:
                    json.dump(run_metadata, mf, indent=2)
            except Exception:
                pass

        # File handles for JSONL
        self._episode_jsonl_handle = open(self.episode_jsonl_path, "a")

    def log_episode(self, episode: int, reward: float, epsilon: float, replay_size: int, moves: int, loss: Optional[float]) -> None:
        record = {
            "episode": episode,
            "reward": reward,
            "epsilon": epsilon,
            "replay": replay_size,
            "moves": moves,
            "loss": loss,
        }

        # JSONL
        self._episode_jsonl_handle.write(json.dumps(record) + "\n")
        self._episode_jsonl_handle.flush()

        # CSV
        with open(self.episode_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward, epsilon, replay_size, moves, loss])

    def log_loss(self, step: int, loss: float) -> None:
        with open(self.loss_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, loss])

    def log_winrate(self, episode: int, ddqn_wins: int, random_wins: int, draws: int, win_rate: float) -> None:
        with open(self.winrate_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, ddqn_wins, random_wins, draws, win_rate])

    def flush(self) -> None:
        try:
            self._episode_jsonl_handle.flush()
        except Exception:
            pass

    def __del__(self):
        try:
            self._episode_jsonl_handle.close()
        except Exception:
            pass


__all__ = ["DDQNMetricWriter"]
