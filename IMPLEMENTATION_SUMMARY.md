# Implementation Summary - Draw Inflation Fixes

## Changes Applied

### 1. Configuration System (`scripts/config_alphazero.py`)

**Added:** `phased_curriculum` configuration with 3 phases

```python
'phased_curriculum': {
    'NUM_ITERATIONS': 100,
    'GAMES_PER_ITERATION': 12,
    'TRAIN_EPOCHS': 10,
    'BATCH_SIZE': 256,
    'BUFFER_SIZE': 5000,
    'phases': [
        # Phase A: Early Exploration (Iter 1-10)
        # Phase B: Balanced Growth (Iter 11-30)
        # Phase C: Full Strength (Iter 31+)
    ]
}
```

**Key Parameters per Phase:**
- MCTS_SIMULATIONS: 400 → 600 → 800
- DIRICHLET_EPSILON: 0.15 → 0.10 → 0.10
- TEMP_THRESHOLD: 15 → 20 → 20
- NO_PROGRESS_PLIES: 60 → 70 → 80
- ENV_MAX_MOVES: 180 → 190 → 200
- MCTS_SEARCH_DRAW_BIAS: -0.06 → -0.05 → -0.03

### 2. MCTS Enhancement (`training/alpha_zero/mcts.py`)

**Added:** Configurable `search_draw_bias` parameter

```python
def __init__(self, ..., search_draw_bias: float = -0.03):
    self.search_draw_bias = float(search_draw_bias)

def _search(self, node, env):
    if env.done and env.winner == 0:
        value = 0.0
        biased_value = value + self.search_draw_bias  # Now configurable
        node.value_sum += value
        node.visits += 1
        return biased_value
```

**Benefits:**
- Allows phase-specific draw aversion during search
- Doesn't affect training targets (which remain 0.0)
- Enables gradual transition from strong to weak draw bias

### 3. Trainer Instrumentation (`training/alpha_zero/trainer.py`)

**Added:** Policy entropy logging

```python
def train_step(self, epochs: int = 1, verbose: bool = True):
    ...
    with torch.no_grad():
        probs = torch.exp(policy_logits)
        entropy = -(probs * policy_logits).sum(dim=1).mean().item()
    print(f"  avg_policy_entropy={entropy:.4f}")
```

**Benefits:**
- Detects if policy is over-committing to safe lines
- High entropy = broad exploration
- Low entropy = confident, specific lines

### 4. Training Script Enhancement (`scripts/train_alphazero.py`)

**Added:** Phase-based parameter application per iteration

```python
for iteration in range(start_iter + 1, CFG['NUM_ITERATIONS'] + 1):
    # Apply phase-specific parameters
    if 'phases' in CFG:
        phase_cfg = get_phase_for_iteration(iteration)
        if phase_cfg:
            mcts.num_simulations = phase_cfg['MCTS_SIMULATIONS']
            mcts.dirichlet_epsilon = phase_cfg['DIRICHLET_EPSILON']
            mcts.search_draw_bias = phase_cfg['MCTS_SEARCH_DRAW_BIAS']
            trainer.temp_threshold = phase_cfg['TEMP_THRESHOLD']
            trainer.env_max_moves = phase_cfg['ENV_MAX_MOVES']
            trainer.no_progress_plies = phase_cfg['NO_PROGRESS_PLIES']
            trainer.draw_penalty = phase_cfg['DRAW_PENALTY']
            mcts.draw_value = phase_cfg['MCTS_DRAW_VALUE']
```

**Benefits:**
- Automatic phase transitions
- Clear logging of active phase
- No manual intervention needed

## How It Works

### Phase A: Early Exploration (Iterations 1-10)

**Goal:** Prevent MCTS from finding deep draw loops

**Mechanisms:**
1. **Weak search (400 sims)**: Doesn't explore deeply enough to find draw basins
2. **High exploration (0.15 Dirichlet)**: Spreads visits beyond first safe option
3. **Early exploitation (temp=15)**: Commits to decisive lines sooner
4. **Short games (180 moves, 60 no-progress)**: Cuts long stalemates
5. **Strong draw bias (-0.06)**: Makes draws unattractive during search

**Expected Outcome:**
- Draw rate: 8% → ~50% (still high, but policy learning decisive patterns)
- Avg length: 81 → ~110 moves
- Policy entropy: High (exploring broadly)

### Phase B: Balanced Growth (Iterations 11-30)

**Goal:** Gradually increase search strength as value head improves

**Mechanisms:**
1. **Moderate search (600 sims)**: Deeper exploration, but not too deep
2. **Standard exploration (0.10 Dirichlet)**: Back to normal noise
3. **Standard exploitation (temp=20)**: Normal temperature schedule
4. **Moderate game length (190 moves, 70 no-progress)**: Balanced
5. **Moderate draw bias (-0.05)**: Reduced but still present

**Expected Outcome:**
- Draw rate: ~50% → ~35%
- Avg length: ~110 → ~100 moves
- Policy entropy: Decreasing (committing to patterns)

### Phase C: Full Strength (Iterations 31+)

**Goal:** Use full MCTS power with well-formed policy/value

**Mechanisms:**
1. **Full search (800 sims)**: Deep exploration for high-quality play
2. **Standard exploration (0.10 Dirichlet)**: Normal noise
3. **Standard exploitation (temp=20)**: Normal temperature schedule
4. **Full game length (200 moves, 80 no-progress)**: No restrictions
5. **Weak draw bias (-0.03)**: Minimal, only for tie-breaking

**Expected Outcome:**
- Draw rate: ~35% → ~25-30% (stable)
- Avg length: ~100 moves (stable)
- Policy entropy: Low (confident, decisive)
- Win rate: >55% (away from random)

## Verification Checklist

- [x] Configuration file updated with phased_curriculum
- [x] MCTS supports configurable search_draw_bias
- [x] Trainer logs policy entropy
- [x] Training script applies phase parameters per iteration
- [x] Phase transitions are automatic
- [x] Documentation created (DRAW_INFLATION_FIXES.md)
- [x] Quick start guide created (PHASED_CURRICULUM_QUICK_START.md)

## Testing

To verify the implementation works:

```bash
# View the phased curriculum config
python scripts/config_alphazero.py phased_curriculum

# Start training (will apply phases automatically)
python scripts/train_alphazero.py --config phased_curriculum

# Monitor metrics in CSV
tail -f data/training_logs/alphazero_training.csv
```

## Expected Behavior

### Iteration 1
```
📋 Phase A: Early Exploration (Iter 1-10)
  MCTS sims: 400, Dirichlet ε: 0.15, Search bias: -0.06
  Temp threshold: 15, Max moves: 180, No-progress: 60
```

### Iteration 10
```
📋 Phase A: Early Exploration (Iter 1-10)
  MCTS sims: 400, Dirichlet ε: 0.15, Search bias: -0.06
  Temp threshold: 15, Max moves: 180, No-progress: 60
```

### Iteration 11
```
📋 Phase B: Balanced Growth (Iter 11-30)
  MCTS sims: 600, Dirichlet ε: 0.10, Search bias: -0.05
  Temp threshold: 20, Max moves: 190, No-progress: 70
```

### Iteration 31
```
📋 Phase C: Full Strength (Iter 31+)
  MCTS sims: 800, Dirichlet ε: 0.10, Search bias: -0.03
  Temp threshold: 20, Max moves: 200, No-progress: 80
```

## Metrics to Track

In `data/training_logs/alphazero_training.csv`:

| Metric | Phase A Target | Phase B Target | Phase C Target |
|--------|---|---|---|
| draw_rate | <60% by iter 5 | ~40-50% | ~25-30% |
| avg_game_length | ~110 | ~100 | ~100 |
| value_loss | Decreasing | <0.1 | <0.05 |
| policy_loss | ~0.7 | ~0.5 | ~0.4 |
| p1_win_rate | ~50% | >50% | >55% |

## Troubleshooting

### Issue: Draw rate not decreasing in Phase A
**Solution:** 
- Increase DIRICHLET_EPSILON to 0.20
- Decrease TEMP_THRESHOLD to 10
- Increase MCTS_SEARCH_DRAW_BIAS to -0.10

### Issue: Games too short
**Solution:**
- Increase NO_PROGRESS_PLIES in Phase A
- Increase ENV_MAX_MOVES in Phase A

### Issue: Training too slow
**Solution:**
- Reduce MCTS_SIMULATIONS in Phase A (try 300)
- Increase GAMES_PER_ITERATION in config

### Issue: Value loss not decreasing
**Solution:**
- Increase TRAIN_EPOCHS in config
- Increase learning rate (currently 0.001)
- Check if buffer has enough diverse data

## Files Modified

1. `scripts/config_alphazero.py` - Added phased_curriculum config
2. `training/alpha_zero/mcts.py` - Added search_draw_bias parameter
3. `training/alpha_zero/trainer.py` - Added policy entropy logging
4. `scripts/train_alphazero.py` - Added phase application logic

## Files Created

1. `DRAW_INFLATION_FIXES.md` - Detailed explanation of fixes
2. `PHASED_CURRICULUM_QUICK_START.md` - Quick reference guide
3. `IMPLEMENTATION_SUMMARY.md` - This file

## Next Steps

1. Run training with phased curriculum
2. Monitor metrics against success criteria
3. Adjust phase boundaries if needed
4. Experiment with phase parameters
5. Log MCTS tree statistics for deeper analysis

## References

- Original analysis: Silent Issue Analysis From Your New Logs
- AlphaZero paper: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
- Curriculum learning: Self-Paced Learning for Mutual Information Maximization
