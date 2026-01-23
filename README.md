# Comparative Analysis of Reinforcement Learning Agents in Checkers
## A Study of D3QN and AlphaZero-Inspired Architectures

**Authors:** Anas Shawki Mtaweh, Riad Al Eid
**Institution:** Lebanese International University (LIU)
**Supervisor:** Dr. Ali Mohamad Ballout
**Date:** January 2026

## Abstract

This project investigates tabula rasa learning in the discrete, adversarial environment of Checkers (8×8). Two fundamentally different deep reinforcement learning paradigms are designed, trained, and evaluated:

*   **D3QN (Dueling Double Deep Q-Network):** A value-based approach emphasizing state evaluation and action selection.

*   **AlphaZero-Inspired Agent:** A model-based approach combining a dual-head neural network with Monte Carlo Tree Search (MCTS).

Experimental results show that while the D3QN agent achieves strong baseline performance (96.2% win rate against a random agent), the AlphaZero-inspired agent demonstrates superior empirical performance, achieving a 0% loss rate in head-to-head evaluations against the best D3QN model under the tested conditions.

## Project Structure

The repository is organized into independent workspaces corresponding to each architecture and interface.

.
├── d3qn_workspace/            # D3QN (Value-Based Learning)
│   ├── core/                  # Game logic (Board, Rules, Move generation)
│   ├── training/              # Network architecture & Replay Buffer
│   └── checkpoints/           # Saved models (e.g., gen7_specialist.pth)
│
├── mcts_workspace/            # AlphaZero-Inspired (Search-Based Learning)
│   ├── core/                  # State encoders & move parsers
│   ├── training/              # MCTS logic, dual-head network, Ray workers
│   ├── scripts/               # Training & evaluation scripts
│   └── checkpoints/           # Model checkpoints (Iter 100–229, Era 9)
│
├── web/                       # Web-based Checkers Interface
│   ├── app.py                 # Flask backend
│   └── static/                # Frontend (JS/CSS)
│
└── evaluation_results/        # CSV files from head-to-head evaluations

## Methodology & Architectures
### 1. D3QN (Value-Based Learning)

The D3QN agent employs a Dueling Double DQN architecture to decouple state value estimation from action advantages.

**Input Representation:**
5-channel spatial encoding (own pieces, opponent pieces, kings, turn context)

**Best Model:**
gen7_specialist.pth

**Training Strategy:**
Iterative reward refinement with terminal-focused reward shaping to prevent reward hacking.

**Key Result:**
96.2% win rate against a random agent at ~18,500 training episodes.

### 2. AlphaZero-Inspired Agent (Policy + Search)

This agent follows the AlphaGo Zero paradigm, combining MCTS with a dual-head neural network predicting policy and value.

**Input Representation:**
6-channel encoding (includes forced-move context)

**Best Model:**
Checkpoints iter_200+ (Era 9)

**Search Configuration:**
1600 MCTS simulations per move

**Key Result:**
0% loss rate against the best D3QN agent in direct evaluations.

Note: This implementation is AlphaZero-inspired, incorporating domain-specific adaptations and custom encodings.

## Results & Benchmarks

Final evaluation compares the strongest D3QN model with the strongest AlphaZero-inspired agent.

| Metric | D3QN (Gen 7) | AlphaZero-Inspired (Era 9) |
| :--- | :--- | :--- |
| Learning Paradigm | Value-Based | Policy + Search (MCTS) |
| Forward Simulations | 0 | 1600 per move |
| Long-Horizon Planning | Limited (>4 moves) | Significantly improved (8+ moves) |
| Win Rate vs Random | 96.2% | 100% |
| Head-to-Head vs D3QN | — | No losses observed |

Figure 1: plot_200_229.png (included in repository) shows the AlphaZero-inspired agent’s stability phase where loss rate converges to 0.0.

## Usage
### 1. Installation

Requirements:

*   Python 3.10+

*   CUDA support recommended (for AlphaZero-inspired training)

```bash
pip install torch numpy flask ray pandas matplotlib
```

### 2. Running the Web Interface

To play against trained agents or observe automated matches:

```bash
python web/app.py
```


Open http://127.0.0.1:5000 in a browser.

Select Player 1 and Player 2 models (e.g., gen7_specialist vs checkpoint_iter_229) and start the game.

### 3. Training the AlphaZero-Inspired Agent

To resume or start training using predefined configurations:

```bash
cd mcts_workspace
python scripts/train_alphazero.py --config era9_precision --workers 4
```

### 4. Model Evaluation

To run a headless tournament between agents:

```bash
cd mcts_workspace
python scripts/evaluate_alphazero_vs_d3qn.py
```

## Hyperparameters
### D3QN (Gen 7)

*   **Optimizer:** Adam (learning rate = 1×10⁻⁴)

*   **Loss Function:** Huber Loss (Smooth L1)

*   **Exploration:** Linear ε-decay (1.0 → 0.05)

*   **Rewards:**

    *   Win: +1.0

    *   Loss: −1.0

    *   Step penalty: −0.001

### AlphaZero-Inspired (Era 9)

*   **MCTS Simulations:** 1600

*   **Cₚᵤ𝒸ₜ:** 1.5

*   **Dirichlet Noise:** ε = 0.15, α = 0.5

*   **Training Batch Size:** 256

*   **Replay Buffer:** 50,000 transitions (FIFO)

## Credits

This project was submitted to the School of Arts & Sciences, Lebanese International University.

**Developers:**
Anas Shawki Mtaweh
Riad Al Eid

**Supervisor:**
Dr. Ali Mohamad Ballout

**Core Libraries:** PyTorch, NumPy, Ray

© 2026 — All Rights Reserved.
