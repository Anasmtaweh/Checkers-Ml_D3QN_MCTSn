# Draw Inflation Fixes - AlphaZero Training

## Overview

This document describes the fixes applied to address the **draw inflation issue** identified in the training logs. The system was transitioning from "copying randomness" to "copying long draw-ish search behavior," resulting in:

- Draw rate rising to 83%
- Average game length increasing to 140+ moves
- Policy loss dominating mid-run (~0.7)
- Iteration time increasing to ~1721s

## Root Causes

### Silent Issue 1: Excessive Early Exploitation
- MCTS with 800 simulations from iteration 1 quickly focuses on long, safe lines
- Policy targets derived from MCTS visits learn to imitate draw-prone distributions
- Result: draw rate balloons before value head can sculpt win/loss signal

### Silent Issue 2: Weak Search-Time Draw Bias
- Training target: draw = -0.05 (correct)
- Search-time bias: -0.03 (too weak to break draw basins with 800 sims)
- Small bias easily overwhelmed by visit priors once tree finds safe loops

### Silent Issue 3: Buffer Dominated by Long Draw Games
- After iteration 4+, majority of transitions are from 129-140 move games
- Signal-to-noise for decisive lines is lower (few wins/losses vs many draws)
- Slows policy improvement

### Silent Issue 4: Low Self-Play Throughput
- 800 sims/move with 12 games = ~20-28 minutes per iteration
- Early training benefits more from broader, noisier coverage
- With long games, total MCTS calls explode

## Applied Fixes

### 1. Phased Curriculum Configuration

A new `phased_curriculum` configuration has been added to `scripts/config_alphazero.py` that implements a three-phase training schedule:

#### Phase A: Early Exploration (Iterations 1-10)
- **MCTS_SIMULATIONS**: 400 (down from 800)
- **DIRICHLET_EPSILON**: 0.15 (up from 0.1)
- **TEMP_THRESHOLD**: 15 (down from 20)
- **NO_PROGRESS_PLIES**: 60 (down from 80)
- **ENV_MAX_MOVES**: 180 (down from 200)
- **DRAW_PENALTY**: -0.05
- **MCTS_DRAW_VALUE**: -0.06
- **MCTS_SEARCH_DRAW_BIAS**: -0.06

**Expected Outcome:**
- Shorter games (~100-115 moves)
- Lower draw rate (~40-55%)
- More decisive targets
- Policy learns win/loss shapes, not just stalling

#### Phase B: Balanced Growth (Iterations 11-30)
- **MCTS_SIMULATIONS**: 600
- **DIRICHLET_EPSILON**: 0.10
- **TEMP_THRESHOLD**: 20
- **NO_PROGRESS_PLIES**: 70
- **ENV_MAX_MOVES**: 190
- **DRAW_PENALTY**: -0.05
- **MCTS_DRAW_VALUE**: -0.05
- **MCTS_SEARCH_DRAW_BIAS**: -0.05

**Expected Outcome:**
- Draw rate drops toward ~30-40%
- Value head gets decisive reinforcement
- Throughput remains manageable

#### Phase C: Full Strength (Iterations 31+)
- **MCTS_SIMULATIONS**: 800
- **DIRICHLET_EPSILON**: 0.10
- **TEMP_THRESHOLD**: 20
- **NO_PROGRESS_PLIES**: 80
- **ENV_MAX_MOVES**: 200
- **DRAW_PENALTY**: -0.05
- **MCTS_DRAW_VALUE**: -0.05
- **MCTS_SEARCH_DRAW_BIAS**: -0.03

**Expected Outcome:**
- Strong policy with decisive tendencies
- Balanced dataset diversity
- High-quality search once policy/value are better formed

### 2. Configurable Search-Time Draw Bias

The MCTS now supports a configurable `search_draw_bias` parameter (previously hardcoded to -0.03):

```python
# In training/alpha_zero/mcts.py
def __init__(self, ..., search_draw_bias: float = -0.03):
    self.search_draw_bias = float(search_draw_bias)

def _search(self, node, env):
    if env.done and env.winner == 0:
        value = 0.0
        biased_value = value + self.search_draw_bias  # Now configurable
        ...
```

This allows phase-specific draw aversion during search without affecting training targets.

### 3. Policy Entropy Logging

Added policy entropy computation during training for diagnostics:

```python
# In training/alpha_zero/trainer.py train_step()
with torch.no_grad():
    probs = torch.exp(policy_logits)
    entropy = -(probs * policy_logits).sum(dim=1).mean().item()
print(f"  avg_policy_entropy={entropy:.4f}")
```

**Interpretation:**
- High entropy: Policy is uncertain, exploring broadly
- Low entropy: Policy is confident, committing to specific lines
- If entropy drops too fast while draw rate rises: Over-committing to safe lines

### 4. Phase-Based Parameter Updates

The training script now applies phase-specific parameters per iteration:

```python
# In scripts/train_alphazero.py
if 'phases' in CFG:
    phase_cfg = None
    for phase in CFG['phases']:
        if phase['iter_start'] <= iteration <= phase['iter_end']:
            phase_cfg = phase
            break
    
    if phase_cfg:
        # Update MCTS and trainer parameters dynamically
        mcts.num_simulations = phase_cfg.get('MCTS_SIMULATIONS', ...)
        mcts.dirichlet_epsilon = phase_cfg.get('DIRICHLET_EPSILON', ...)
        mcts.search_draw_bias = phase_cfg.get('MCTS_SEARCH_DRAW_BIAS', ...)
        trainer.temp_threshold = phase_cfg.get('TEMP_THRESHOLD', ...)
        # ... etc
```

## Usage

### Start Training with Phased Curriculum

```bash
python scripts/train_alphazero.py --config phased_curriculum
```

### Resume Training

```bash
python scripts/train_alphazero.py --config phased_curriculum --resume 10
```

### Use Standard (Non-Phased) Config

```bash
python scripts/train_alphazero.py --config standard
```

## Success Criteria

Monitor these metrics to confirm the fixes are working:

### Draw Rate Trajectory
- **Iter 1-5**: <60% by iter 5
- **Iter 6-10**: ~40-50%
- **Iter 11-20**: ~30-40%

### Average Game Length
- **<120 moves by iter 10**

### Value Loss
- **Continues downward trend (<0.1 by iter 12-15)**

### Policy Loss
- **Declines and stabilizes (~0.4-0.6 later)**

### Policy Entropy
- **Decreases steadily; if it drops too fast while draw rate rises, you're over-committing**

### Win Rate
- **Moves away from random oscillation: >55% by iter 15**

## Files Modified

1. **scripts/config_alphazero.py**
   - Added `phased_curriculum` configuration with three phases
   - Each phase specifies MCTS, environment, and training parameters

2. **training/alpha_zero/mcts.py**
   - Added `search_draw_bias` parameter to `__init__`
   - Updated `_search()` to use configurable `self.search_draw_bias`

3. **training/alpha_zero/trainer.py**
   - Added `dirichlet_epsilon` parameter to `__init__`
   - Added policy entropy logging in `train_step()`

4. **scripts/train_alphazero.py**
   - Added phase-based parameter update logic in training loop
   - Dynamically applies phase-specific MCTS and trainer settings per iteration
   - Prints phase information at start of each iteration

## Instrumentation for Monitoring

The system now logs:
- **Policy entropy**: Indicates if policy is over-committing to safe lines
- **MCTS parameters per iteration**: Shows which phase is active
- **Game statistics**: Draw rate, avg length, win rates
- **Loss metrics**: Value loss, policy loss, total loss

All metrics are logged to `data/training_logs/alphazero_training.csv` for analysis.

## Theoretical Justification

### Why Phased Curriculum Works

1. **Early Phase (Weak Search)**: Broader exploration prevents premature convergence to draw basins
2. **Increased Dirichlet Noise**: Spreads visits beyond first safe basin
3. **Shorter Horizon**: NO_PROGRESS_PLIES=60 cuts long stalemates
4. **Stronger Draw Bias**: -0.06 search bias makes draws less attractive during search
5. **Earlier Exploitation**: temp_threshold=15 commits to decisive lines sooner

2. **Middle Phase (Balanced)**: Gradually increase search strength as value head improves
3. **Late Phase (Full Strength)**: Once policy/value are well-formed, use full MCTS power

### Why This Breaks Draw Basins

- **Curriculum prevents lock-in**: Early weak search doesn't find deep draw loops
- **Decisive targets early**: Policy learns win/loss patterns before draw patterns dominate
- **Value head guidance**: As value head improves, MCTS naturally avoids draw-heavy lines
- **Gradual transition**: Smooth progression prevents sudden distribution shifts

## Assumptions

- Environment's terminal detection is correct
- MCTS backprop draw bias was -0.03 (now configurable)
- Ray worker GPU allocation is not throttling
- Increasing iteration time is due to longer games, not cluster instability

## Next Steps

1. **Run phased curriculum training** for 30+ iterations
2. **Monitor metrics** against success criteria
3. **Adjust phase boundaries** if needed (e.g., extend Phase A to iter 15 if draw rate still high)
4. **Experiment with phase parameters** if results don't match expectations
5. **Log MCTS tree statistics** (depth, Q-value histogram) for deeper analysis

## References

- Original analysis: Silent Issue Analysis From Your New Logs
- AlphaZero paper: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
- Curriculum learning: Self-Paced Learning for Mutual Information Maximization
