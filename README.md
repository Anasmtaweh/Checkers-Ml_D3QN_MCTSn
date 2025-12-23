# ML_Gen2 - Checkers AI

Advanced checkers AI using D3QN (Deep Dueling Q-Network) and MCTS (Monte Carlo Tree Search) algorithms.

## Project Structure

```
ML_Gen2/
├── agents/                          # Pre-trained AI agents
│   └── d3qn/                        # D3QN model checkpoints
│       ├── gen8_titan_LEGACY.pth    # Best performer (68.8%)
│       ├── gen11_ep500_80vR_75vT_CHAMPION.pth
│       └── ...
├── core/                            # Game engine & core logic
│   ├── game.py                      # CheckersEnv (game simulator)
│   ├── board.py                     # Board state management
│   ├── rules.py                     # Checkers rules validation
│   ├── action_manager.py            # Action encoding/decoding
│   ├── board_encoder.py             # State feature extraction
│   └── move_parser.py               # Move notation parsing
├── training/                        # Training pipelines
│   ├── d3qn/                        # D3QN training code
│   │   ├── model.py                 # D3QN neural network architecture
│   │   ├── trainer.py               # Training loop & utilities
│   │   ├── agent.py                 # D3QN agent for self-play
│   │   ├── buffer.py                # Replay buffer implementation
│   │   └── self_play.py             # Self-play training
│   └── mcts/                        # MCTS search implementation
│       ├── mcts_node.py             # MCTS tree node
│       └── mcts_agent.py            # MCTS search agent
├── evaluation/                      # Testing & benchmarking
│   ├── play_vs_mcts.py              # MCTS vs D3QN gauntlet
│   ├── tournament.py                # Round-robin tournaments
│   ├── benchmark.py                 # Benchmark vs random agent
│   └── evaluate_agent.py            # Agent evaluation
├── scripts/                         # Entry points & utilities
│   ├── train_d3qn.py                # Main training script
│   ├── check_checkpoints.py         # Model inspection
│   └── iron_tournament.py           # Swiss tournament system
├── data/                            # Training data & results
│   ├── training_logs/               # CSV training records
│   ├── tournament_results/          # Tournament statistics
│   └── archives/                    # Historical model checkpoints
├── utils/                           # Utility functions
│   └── plot_logs.py                 # Training visualization
├── web/                             # Web interface
│   ├── app.py                       # Flask server
│   ├── index.html                   # Web UI
│   ├── style.css                    # Styling
│   └── game.js                      # Game visualization
└── docs/                            # Documentation
    └── DQN_PROJECT_SUMMARY.md       # Project overview
```

## Quick Start

### Play Against MCTS
```bash
python evaluation/play_vs_mcts.py
```

### Run Tournament
```bash
python evaluation/tournament.py
```

### Train New Agent
```bash
python scripts/train_d3qn.py --episodes 1000 --checkpoint-freq 100
```

## Model Performance

| Model | Type | Tournament | vs Random |
|-------|------|-----------|-----------|
| gen8_titan_LEGACY | D3QN Gen8 | **68.8%** | - |
| gen11_CHAMPION | D3QN Gen11 | 62.5% | 80% |
| gen12_elite_3500 | D3QN Gen12 | 54.2% | 77% |
| MCTS (7s per move) | Tree Search | TBD | - |

## Key Improvements Applied

### MCTS Enhancements
- **Neural Network Evaluation**: Uses champion D3QN for position evaluation
- **Move Ordering**: Prioritizes capture moves for efficient search
- **Time Limit**: 7 seconds per move with adaptive depth
- **Exploration**: UCB weight of 2.0 for aggressive play
- **Rollout Depth**: Extended to 50 plies with advanced heuristics

### D3QN Architecture
- Dueling architecture: Value + Advantage streams
- Convolutional feature extraction (5→32→64→64 channels)
- Layer normalization for stability
- Per-side head support for multi-player evaluation

## Files Modified for MCTS Strength

- `training/mcts/mcts_node.py` - Advanced evaluation with neural integration
- `training/mcts/mcts_agent.py` - Extended time budget and configuration
- `evaluation/play_vs_mcts.py` - Model loader with neural evaluator

## References

- Dueling DQN: https://arxiv.org/abs/1511.06581
- MCTS: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
- Checkers Rules: https://www.fda.org.uk/
