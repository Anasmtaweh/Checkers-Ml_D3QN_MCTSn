# Checkers AI: AlphaZero vs. D3QN

**ML_Gen2** is a deep reinforcement learning project focused on mastering the game of Checkers (English Draughts). It implements and benchmarks two distinct architectures: **AlphaZero** (Model-Based) and **D3QN** (Model-Free).

The project features a complete training pipeline, an automated tournament framework, and a web-based interface for visualizing the agents' thought processes.

## 🌟 Key Features

* **AlphaZero Implementation:** Custom "Tabula Rasa" learning engine using Monte Carlo Tree Search (MCTS) guided by a Dual-Head ResNet.
* **D3QN Specialist:** A robust baseline agent using Dueling Double Deep Q-Networks with prioritized experience replay.
* **Performance:** The AlphaZero agent (Gen 9) achieved a **0% Loss Rate** against the D3QN baseline ITER 220 --> 229 self-play.
* **Web Interface:** A Flask-based UI allowing humans to play against the AI or watch "Brain vs Brain" matches with real-time probability visualization.
* **Evaluation Suite:** Automated scripts to run head-to-head tournaments and generate win-rate plots.

---

## 📊 Performance Results

We benchmarked the **AlphaZero (Gen 9)** agent against our strongest **D3QN (Gen 7)** specialist.

| Agent | Architecture | Training Method | Win Rate | Loss Rate |
| --- | --- | --- | --- | --- |
| **D3QN (Gen 7)** | Model-Free (Value Based) | Q-Learning | 0.0% | 17.0% |
| **AlphaZero (Gen 9)** | Model-Based (MCTS) | Self-Play | **17.0%** | **0.0%** |

*Note: The remaining games ended in Draws, which is typical for perfect play in Checkers.*

---

## 📂 Project Structure

* **`mcts_workspace/`**: The Core AlphaZero Engine.
    * `core/`: Game rules, bitboard operations, and move validation.
    * `training/`: Neural network (PyTorch) and MCTS logic.
* **`d3qn_workspace/`**: The Baseline D3QN Agent.
* **`web/`**: Flask application for the browser interface.
* **`evaluation_results/`**: Tournament logs and matplotlib graphs.

---

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Anasmtaweh/Checkers-Ml_D3QN_MCTSn.git
cd Checkers-Ml_D3QN_MCTSn
```

2. **Install Dependencies:**
The project requires Python 3.8+ and PyTorch.
```bash
pip install torch numpy flask pandas matplotlib tqdm
```

*(Note: CUDA is recommended for training, but CPU works for inference.)*

---

## 🚀 Usage

### 1. Play vs AI (Web Interface)

Launch the interactive game board:

```bash
python web/app.py
```

Open your browser to `http://127.0.0.1:5000`. You can select "Human vs AlphaZero" or watch "AlphaZero vs D3QN".

### 2. Train from Scratch

To start a new training run for AlphaZero:

```bash
python mcts_workspace/scripts/train_alphazero.py --config madras_local_resume
```

*Logs are saved to `mcts_workspace/data/training_logs/`.*

### 3. Run Tournament

To verify agent strength:

```bash
python mcts_workspace/scripts/evaluate_alphazero_vs_d3qn.py
```

---

## 👥 Authors

* **Anas Shawki Mtaweh**

**Supervised by:** Dr. Ali Mohamad Ballout
*Lebanese International University*

---

## 📄 License

This project is open-source and available under the MIT License.
