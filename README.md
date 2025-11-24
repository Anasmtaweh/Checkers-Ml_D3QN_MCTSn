# 🎮 Dama ML — Reinforcement Learning Agents for Checkers

## Project Overview
Dama ML is a checkers (Dama) reinforcement learning sandbox. It includes a custom environment with mandatory multi-jump captures, three agent types (random, Q-learning RL, heuristic hybrid), CLI match control, training scripts, and a Flask web UI for visualizing games. Turns, capture chains, and king promotion are enforced in the environment.

## Features
- Multiple agents: `random`, `rl` (Q-table), `hybrid` (heuristics + greedy + Q-table scoring)
- CLI-controlled matches with color-coded sides (Red = Player 1, White = Player 2)
- Scoreboard in web UI: agent identities, piece counts, king counts, current turn
- Mandatory capture rules with recursive multi-jump chains
- Q-learning persistence (loads legacy `q_table.pkl` or saves to `models/`)
- Config-driven defaults (`config_agents.json`)
- Modular code: env, agents, web UI, training scripts, tests

## Folder Structure
```
dama-ml/
├── agents/
│   ├── hybrid_agent.py
│   ├── random_agent.py
│   └── rl_agent.py
├── dama_env/
│   ├── board.py
│   ├── env.py
│   └── rules.py
├── web/
│   ├── app.py
│   ├── static/
│   │   ├── board.css
│   │   ├── board.js
│   │   └── style.css   # legacy; index.html uses board.css/board.js
│   └── templates/
│       └── index.html
├── config_agents.json
├── play_one_game.py
├── train_rl.py
├── train_hybrid.py
├── watch_trained.py
├── test_env.py
├── test_env_run.py
├── test_multijump.py
├── test_hybrid.py
├── q_table.pkl                 # legacy model fallback
├── flask_test_run.py
└── models/                     # created by training scripts (rl_q_table.pkl, hybrid_model.pkl)
```
*Note: `models/` is expected to hold trained artifacts; create it if absent before saving models.*

## Installation (Ubuntu / Python 3.12)
```bash
git clone <repo> dama-ml
cd dama-ml
python3 -m venv .venv
source .venv/bin/activate
pip install flask numpy
```

## Running the Flask Web App
```bash
source .venv/bin/activate
cd web
python app.py
```
Open http://127.0.0.1:5000 to see the board, agent labels, piece/king counts, and turn indicator.

## Playing Matches from CLI
`play_one_game.py` supports agent selection per color (defaults from `config_agents.json`, red=rl, white=random):
```bash
python play_one_game.py --red=rl --white=hybrid
python play_one_game.py --red=random --white=rl
```
Terminal output shows identities, turn order, and multi-jump continuity. Red = Player 1, White = Player 2; kings are highlighted in the environment render.

## Training Commands
RL agent training (Q-learning):
```bash
python train_rl.py --opponent=random --episodes=2000
python train_rl.py --opponent=hybrid --episodes=2000
python train_rl.py --opponent=rl --episodes=2000
```
Hybrid training (same loop, stores table for heuristic scoring):
```bash
python train_hybrid.py --opponent=random --episodes=3000
python train_hybrid.py --opponent=rl --episodes=3000
python train_hybrid.py --opponent=hybrid --episodes=3000
```
- Models save to `models/rl_q_table.pkl` or `models/hybrid_model.pkl` (directory is created by the scripts; create manually if permissions require).
- Delete the files in `models/` (or `q_table.pkl` legacy) to reset training.
- Opponent types: `random`, `hybrid`, or fixed-policy `rl`.

## Agent Descriptions
- **RandomAgent** (`agents/random_agent.py`): uniform random legal move.
- **QLearningAgent / RLAgent** (`agents/rl_agent.py`): Q-table on atomic steps; handles multi-jump by queuing capture steps. `RLAgent` loads a table and does not learn.
- **HybridAgent** (`agents/hybrid_agent.py`): heuristic ordering (mandatory captures, promotion, danger avoidance) with Q-table scoring fallback.

## Multi-Jump Logic
- In `dama_env/rules.py`, capture sequences are generated recursively as lists of `(start, landing, jumped)` triples.
- `dama_env/env.py` applies capture chains step-by-step; the same player continues during a chain. Turn switches only when no further capture is available or after a non-capturing move.
- Rewards: +10 per captured piece, optional -1 per move, +50 win / -50 loss.
- Kings (±2) move both directions; promotion handled via board state.

## Web UI (Scoreboard + Colors)
- Canvas board with red (Player 1) and white (Player 2) pieces; kings outlined in gold.
- Scoreboard (from backend responses) shows: agent names for Red/White, remaining pieces, kings, and current turn.
- Buttons trigger predefined matchups and step-through play. No training occurs in the UI; it only loads existing models.

## Model Files
- Legacy fallback: `q_table.pkl` (root).
- Preferred locations (created by training scripts): `models/rl_q_table.pkl`, `models/hybrid_model.pkl`.

## Config File
`config_agents.json` sets default CLI agents:
```json
{
  "red": "rl",
  "white": "random"
}
```
CLI flags override these defaults.

## Future Improvements
- Stronger heuristics and danger evaluation
- Deep RL policies (DQN/Actor-Critic)
- Richer GUI with animations/highlighting
- Opening book or endgame tablebases
- Analytics dashboard for training and match stats
