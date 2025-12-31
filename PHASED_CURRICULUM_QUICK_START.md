# Quick Start Guide - Phased Curriculum Training

## TL;DR

To fix draw inflation and train with the phased curriculum:

```bash
python scripts/train_alphazero.py --config phased_curriculum
```

## What Changed

The system now uses a **3-phase curriculum** to prevent draw inflation:

| Phase | Iterations | MCTS Sims | Dirichlet ε | Temp Threshold | Max Moves | No-Progress | Search Bias |
|-------|-----------|-----------|------------|----------------|-----------|------------|------------|
| A (Early) | 1-10 | 400 | 0.15 | 15 | 180 | 60 | -0.06 |
| B (Balanced) | 11-30 | 600 | 0.10 | 20 | 190 | 70 | -0.05 |
| C (Full) | 31+ | 800 | 0.10 | 20 | 200 | 80 | -0.03 |

## Expected Results

### Phase A (Iter 1-10)
- Draw rate: 8% → ~50%
- Avg game length: 81 → ~110 moves
- Iteration time: ~10-15 min (faster due to fewer sims)

### Phase B (Iter 11-30)
- Draw rate: ~50% → ~35%
- Avg game length: ~110 → ~100 moves
- Iteration time: ~15-20 min

### Phase C (Iter 31+)
- Draw rate: ~35% → ~25-30%
- Avg game length: ~100 moves (stable)
- Iteration time: ~20-25 min
- Win rate: >55%

## Key Metrics to Monitor

Open `data/training_logs/alphazero_training.csv` and watch:

1. **draw_rate**: Should decrease over phases
2. **avg_game_length**: Should stabilize <120 moves
3. **p1_win_rate**: Should move away from 50% (random)
4. **value_loss**: Should continue decreasing
5. **policy_loss**: Should stabilize around 0.4-0.6

## Commands

### Start Fresh
```bash
python scripts/train_alphazero.py --config phased_curriculum
```

### Resume from Iteration 10
```bash
python scripts/train_alphazero.py --config phased_curriculum --resume 10
```

### Use Standard (Non-Phased) Config
```bash
python scripts/train_alphazero.py --config standard
```

### View Configuration
```bash
python scripts/config_alphazero.py phased_curriculum
```

## What's Different from Before

### Before (Standard Config)
- MCTS: 800 sims from iteration 1
- Dirichlet: 0.1 (less exploration)
- Temp threshold: 20 (later exploitation)
- Max moves: 200 (long games allowed)
- No-progress: 80 (long stalemates allowed)
- Search draw bias: -0.03 (weak)

### After (Phased Curriculum)
- **Phase A**: Weak search (400 sims) + strong exploration (0.15 Dirichlet) + early exploitation (temp=15) + short games (180 moves, 60 no-progress) + strong draw bias (-0.06)
- **Phase B**: Gradual increase (600 sims, 0.10 Dirichlet, 190 moves, 70 no-progress, -0.05 bias)
- **Phase C**: Full strength (800 sims, 0.10 Dirichlet, 200 moves, 80 no-progress, -0.03 bias)

## Why This Works

1. **Early weak search**: Prevents MCTS from finding deep draw loops
2. **High exploration**: Spreads visits beyond first safe basin
3. **Early exploitation**: Commits to decisive lines sooner
4. **Short games**: Cuts long stalemates
5. **Strong draw bias**: Makes draws unattractive during search
6. **Gradual transition**: Smooth progression as value head improves

## Troubleshooting

### Draw rate still high after Phase A?
- Extend Phase A to iteration 15
- Increase DIRICHLET_EPSILON to 0.20
- Decrease TEMP_THRESHOLD to 10

### Games too short?
- Increase NO_PROGRESS_PLIES in Phase A
- Increase ENV_MAX_MOVES in Phase A

### Training too slow?
- Reduce MCTS_SIMULATIONS in Phase A (try 300)
- Increase GAMES_PER_ITERATION in config

### Value loss not decreasing?
- Increase TRAIN_EPOCHS in config
- Increase learning rate (currently 0.001)

## Files to Check

- **Config**: `scripts/config_alphazero.py` (phased_curriculum section)
- **MCTS**: `training/alpha_zero/mcts.py` (search_draw_bias parameter)
- **Trainer**: `training/alpha_zero/trainer.py` (policy entropy logging)
- **Training script**: `scripts/train_alphazero.py` (phase application logic)
- **Logs**: `data/training_logs/alphazero_training.csv` (metrics)

## Documentation

For detailed explanation of the fixes, see: `DRAW_INFLATION_FIXES.md`
