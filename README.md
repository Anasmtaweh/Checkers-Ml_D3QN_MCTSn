# CHECKERS-ML — Deep RL for Checkers

CHECKERS-ML trains and evaluates a Checkers agent using a Dueling Double DQN backbone. The project includes a full game environment with mandatory capture rules, a self‑play trainer with replay buffer, evaluation scripts, and a lightweight web UI for visualizing games.

## What’s inside
- **Environment (`checkers_env/`)**: Rule-accurate 8×8 Checkers with forced captures and multi-jump chains (`env.py`, `rules.py`, `board.py`).
- **Agents (`checkers_agents/`)**: Inference-only DDQN agent and a random baseline. The DDQN agent mirrors trainer masking/normalization logic and returns env-format moves.
- **Training (`training/`)**: Common utilities (board encoder, action manager, replay buffer) plus DDQN model, trainer, metrics, and evaluation helpers.
- **Scripts**: `train.py` (main entry), evaluation utilities, and a Flask web viewer under `web/`.
- **Artifacts**: `models/ddqn/` for checkpoints, `logs/ddqn/` for metrics and plots.

## Core design
- **State encoding**: 12-channel tensor (pieces, kings, side-to-move, capture flag, repetition/move counters, last move).
- **Action space**: Fixed 4,032 actions = all ordered origin→destination squares (64×63). Masks restrict Q-values to legal moves each turn. Captures are mandatory; when any capture exists, quiet moves are dropped. Multi-jump chains are executed step-by-step; the first hop is chosen, and the environment forces continuation via `info["continue"]`.
- **Rewards (env)**: −1 per move, +10 per captured piece, ±50 win/loss bonus. Forced capture continuation is surfaced through `info["continue"]` and `force_capture_from`.
- **Model**: Dueling DDQN CNN with separate online/target nets; double-Q target uses online argmax and target evaluation. Optional hard or soft target updates, gradient clipping, and optional Q clipping.
- **Replay**: Uniform replay buffer storing states, actions, rewards (from acting side), dones, and legal masks for current/next states.

## Quickstart
### Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# For CUDA builds, install the matching torch wheel per https://pytorch.org/get-started/locally/
pip install torch flask pandas matplotlib numpy
```

### Train
Use the tuned stable preset for Checkers:
```bash
python train.py --preset stable-checkers --device cuda
```
Other useful flags:
- `--train-episodes`: total episodes (stable preset defaults to 20,000)
- `--max-moves`: per-game cap (default 200)
- `--epsilon-start/end/decay-steps`: exploration schedule
- `--soft-update/--soft-tau`: enable soft target updates (stable preset: tau=0.005)
- `--save-interval` / `--eval-interval`: checkpointing and periodic eval vs random

Checkpoints land in `models/ddqn/` (`checkpoint_*.pt`, `best_model.pt`, `final.pt`). Metrics and plots go to `logs/ddqn/`.

### Evaluate
Quick win-rate vs random:
```bash
python -m training.ddqn.evaluation --help  # see options
```
Or run the tournament helper to pit multiple checkpoints:
```bash
python -m training.ddqn.tournament_eval --checkpoints "models/ddqn/*.pt" --episodes 20
```

### Web viewer
Launch the Flask UI to step through games:
```bash
cd web
python app.py
# open http://127.0.0.1:5000
```

## Key components (by path)
- `checkers_env/env.py`: Env loop, reward logic, forced capture handling.
- `checkers_env/rules.py`: Move generation (simple and capture sequences).
- `training/common/action_manager.py`: 4,032-action mapping, masks.
- `training/common/board_encoder.py`: 12-plane tensor encoder.
- `training/common/replay_buffer.py`: Replay with legal masks.
- `training/ddqn/network.py`: Dueling CNN.
- `training/ddqn/model.py`: Online/target wrapper, target updates.
- `training/ddqn/trainer.py`: Self-play loop, epsilon-greedy with masking, double-Q update.
- `checkers_agents/ddqn_agent.py`: Inference-time action selection using the same masking as the trainer.

## Notes & tips
- Captures are mandatory; if `info["continue"]` is true the same player must continue the jump chain.
- Reward shaping is confined to the environment rewards; no extra bonuses are injected in training.
- Use `--preset debug` for quick sanity checks, `--preset stable-checkers` for longer runs.
- If you load checkpoints manually, prefer `weights_only=True` with `torch.load` when available to avoid pickle warnings.
