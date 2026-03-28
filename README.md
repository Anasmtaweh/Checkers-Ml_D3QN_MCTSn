<div align="center">

# Checkers RL Oracle

**The tabula rasa, dual-architecture reinforcement learning engine for Checkers.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Powered by PyTorch](https://img.shields.io/badge/Powered%20by-PyTorch-red)](https://pytorch.org)
[![Powered by Flask](https://img.shields.io/badge/Powered%20by-Flask-green)](https://flask.palletsprojects.com/)

</div>

---

## What is the Checkers RL Oracle?
This project is a deep investigation into *tabula rasa* learning in the discrete, adversarial environment of Checkers (8×8). Two fundamentally different deep reinforcement learning paradigms are designed, trained, and evaluated head-to-head: a value-based D3QN agent and a search-based AlphaZero-inspired model.

No human game logs. No opening books. Just pure reinforcement learning.

## Features
- **D3QN (Dueling Double Deep Q-Network)** — A 5-channel spatial encoded, value-based agent that decouples state value estimation from action advantages.
- **AlphaZero-Inspired Engine** — A dual-head neural network combined with Monte Carlo Tree Search (MCTS) evaluating 1,600 simulations per move.
- **Flask Web Arena** — A full browser-based interface to spar interactively against trained agents.
- **Head-to-Head Tournaments** — Headless evaluation scripts to pit neural networks against one another in massive automated tournaments.
- **Ray Distributed Training** — Multi-worker, highly-parallel asynchronous experience collection for AlphaZero.
- **Advanced Reward Shaping** — Terminal-focused reinforcement with living-tax penalties to prevent reward hacking.

## Quick Start

### Prerequisites
- Python 3.10+
- An NVIDIA GPU (recommended for AlphaZero training) or CPU fallback

### Installation
```bash
# 1. Clone the repository (if not already local)
git clone https://github.com/Anasmtaweh/checkers-machine-learning.git
cd ML_Gen2

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Web Interface
python web/app.py
```

### First Run
- Open your browser at `http://127.0.0.1:5000`
- Select Player 1 and Player 2 models (e.g., `gen7_specialist` vs `checkpoint_iter_229`)
- Start the game and watch the agents play (or play against them yourself!)

## Architecture
```text
ML_Gen2/
├── d3qn_workspace/            # D3QN (Value-Based Learning)
│   ├── core/                  # Game logic (Board, Rules, Move generation)
│   ├── training/              # Network architecture & Replay Buffer
│   └── checkpoints/           # Saved models (e.g., gen7_specialist.pth)
│
├── mcts_workspace/            # AlphaZero-Inspired (Search-Based Learning)
│   ├── core/                  # State encoders & move parsers
│   ├── training/              # MCTS logic, dual-head network, Ray workers
│   └── scripts/               # Training & evaluation scripts
│
├── web/                       # Flask Web Arena
│   ├── app.py                 # Backend API and UI orchestrator
│   └── static/                # Frontend Assets (JS/CSS)
│
└── scripts/                   # Utility and testing scripts
```

## Performance & Benchmarks
Final evaluation compares the strongest D3QN model against the strongest AlphaZero-inspired agent. Experimental results show the AlphaZero-inspired agent achieving a `0%` loss rate in head-to-head evaluation against the best D3QN agent.

| Metric                 | D3QN (Gen 7)       | AlphaZero-Inspired (Era 9)         |
| :--------------------- | :----------------- | :--------------------------------- |
| **Learning Paradigm**  | Value-Based        | Policy + Search (MCTS)             |
| **Forward Simulations**| 0                  | 1600 per move                      |
| **Long-Horizon**       | Limited (>4 moves) | Significantly improved (8+ moves)  |
| **Win Rate vs Random** | 96.2%              | 100%                               |
| **Head-to-Head**       | —                  | No losses observed against D3QN    |

## CLI Usage
```bash
# Start the web interface
python web/app.py

# Train the AlphaZero-Inspired Agent
cd mcts_workspace
python scripts/train_alphazero.py --config era9_precision --workers 4

# Run Head-to-Head tournament evaluations
cd mcts_workspace
python scripts/evaluate_alphazero_vs_d3qn.py
```

## Hyperparameters
- **D3QN (Gen 7):** Optimizer: Adam (1×10⁻⁴) | Loss: Huber (Smooth L1) | ε-decay: 1.0 → 0.05
- **AlphaZero (Era 9):** MCTS Sims: 1600 | Cₚᵤ𝒸ₜ: 1.5 | Dirichlet Noise: ε=0.15, α=0.5 | Batch: 256 | Replay: 50k FIFO

## License
MIT — see [LICENSE](LICENSE) for details.

## Author

Built by [Anas Shawki Mtaweh](https://github.com/Anasmtaweh) 

- GitHub: [Anasmtaweh](https://github.com/Anasmtaweh)
- LinkedIn: [linkedin.com/in/anas-mtaweh-a02806218](https://www.linkedin.com/in/anas-mtaweh-a02806218)
