import csv
import json
import os
from typing import Optional, List


class DDQNMetricWriter:
    def __init__(self, base_dir: str = "logs/ddqn", run_metadata: Optional[dict] = None):
        self.base_dir = base_dir
        self.metrics_dir = os.path.join(base_dir, "metrics")
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)

        self.episode_jsonl_path = os.path.join(self.metrics_dir, "episode_stats.jsonl")
        self.episode_path = os.path.join(self.base_dir, "episode_stats.csv")
        self.loss_path = os.path.join(self.base_dir, "loss.csv")
        self.winrate_path = os.path.join(self.metrics_dir, "winrate.csv")

        # Episode stats CSV (append, add header if new)
        write_header = not os.path.exists(self.episode_path)
        self.episode_file = open(self.episode_path, "a", newline="")
        self.episode_writer = csv.writer(self.episode_file)
        if write_header:
            self.episode_writer.writerow(["episode", "reward", "epsilon", "replay", "moves", "loss", "average_reward_50"])

        # Loss CSV
        write_header = not os.path.exists(self.loss_path)
        self.loss_file = open(self.loss_path, "a", newline="")
        self.loss_writer = csv.writer(self.loss_file)
        if write_header:
            self.loss_writer.writerow(["step", "loss"])

        # Winrate CSV
        write_header = not os.path.exists(self.winrate_path)
        self.winrate_file = open(self.winrate_path, "a", newline="")
        self.winrate_writer = csv.writer(self.winrate_file)
        if write_header:
            self.winrate_writer.writerow(["episode", "wins", "losses", "draws", "winrate"])

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
        self.reward_buffer: List[float] = []

    def log_episode(self, episode: int, reward: float, epsilon: float, replay_size: int, moves: int, loss: Optional[float]) -> None:
        record = {
            "episode": episode,
            "reward": reward,
            "epsilon": epsilon,
            "replay": replay_size,
            "moves": moves,
            "loss": loss,
        }

        # running average reward (window 50)
        self.reward_buffer.append(reward)
        if len(self.reward_buffer) > 50:
            self.reward_buffer.pop(0)
        avg_reward = sum(self.reward_buffer) / len(self.reward_buffer)

        # JSONL
        self._episode_jsonl_handle.write(json.dumps(record) + "\n")
        self._episode_jsonl_handle.flush()

        # CSV
        self.episode_writer.writerow([episode, reward, epsilon, replay_size, moves, loss, avg_reward])
        self.episode_file.flush()

    def log_loss(self, step: int, loss: float) -> None:
        self.loss_writer.writerow([step, loss])
        self.loss_file.flush()

    def log_winrate(self, episode: int, ddqn_wins: int, random_wins: int, draws: int, win_rate: float) -> None:
        self.winrate_writer.writerow([episode, ddqn_wins, random_wins, draws, win_rate])
        self.winrate_file.flush()

    def flush(self) -> None:
        try:
            self._episode_jsonl_handle.flush()
        except Exception:
            pass

    def __del__(self):
        try:
            self._episode_jsonl_handle.close()
            self.episode_file.close()
            self.loss_file.close()
            self.winrate_file.close()
        except Exception:
            pass


__all__ = ["DDQNMetricWriter"]
