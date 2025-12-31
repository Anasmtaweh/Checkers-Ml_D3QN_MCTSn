# AlphaZero Checkers - Quick Fix Guide

## The Problem in One Sentence
**Your network is learning to copy random MCTS behavior instead of improving upon it, creating a negative feedback loop that prevents any real learning.**

---

## The 4 Critical Silent Issues

### 1. **Value Loss Weight is 6.7x Too Low**
- Current: `value_loss_weight=0.15` vs `policy_loss_weight=1.0`
- Result: Value head gets starved for gradient signal
- Evidence: `value_loss=1.1813` (very high, network not learning)
- Fix: Set both to `1.0`

### 2. **Draw Values Are Inconsistent**
- Config says: `DRAW_PENALTY=0.0, MCTS_DRAW_VALUE=0.0`
- Network outputs: Tanh range [-1, 1]
- Training targets: 0.0 for draws
- Result: Network can't distinguish draws from random noise
- Fix: Set both to `-0.05` (slight penalty)

### 3. **Replay Buffer Is 95% Stale Data**
- Buffer: 50,000 transitions
- New data per iteration: ~1,200 transitions
- Buffer age: ~42 iterations old
- Result: Network trains on garbage from early iterations
- Fix: Reduce buffer to 5,000, batch to 256

### 4. **Dirichlet Noise Drowns Out Network**
- Current: `dirichlet_alpha=0.6, epsilon=0.25`
- Result: 25% of MCTS priors replaced with random noise
- Impact: Network's policy is completely ignored
- Fix: Reduce to `alpha=0.3, epsilon=0.1`

---

## The Negative Feedback Loop

```
Iteration 1:
  Network = Random
  MCTS = Random (because network is random)
  Network learns to copy random MCTS
  
Iteration 2:
  Network = Slightly better at copying randomness
  MCTS = Still random (because network still outputs random policy)
  Network learns to copy randomness better
  
Iteration 3+:
  Network = Stuck copying randomness
  MCTS = Still random
  [LOOP - STUCK]
```

---

## Immediate Fixes (Apply All 4)

### Fix 1: Update `scripts/config_alphazero.py`

```python
# STANDARD config - CHANGE THESE LINES:

# Line ~30: Change draw values
'DRAW_PENALTY': -0.05,      # Was 0.0
'MCTS_DRAW_VALUE': -0.05,   # Was 0.0

# Line ~35: Reduce buffer size
'BUFFER_SIZE': 5000,        # Was 50000
'BATCH_SIZE': 256,          # Was 512
```

### Fix 2: Update `training/alpha_zero/trainer.py`

```python
# Line ~45: Increase value loss weight
value_loss_weight=1.0,      # Was 0.15
policy_loss_weight=1.0,     # Keep as is

# Line ~130: Reduce Dirichlet noise
mcts = MCTS(
    ...
    dirichlet_alpha=0.3,    # Was 0.6
    dirichlet_epsilon=0.1,  # Was 0.25
)

# Line ~150: Stronger regularization
optimizer = optim.Adam(
    self.model.network.parameters(),
    lr=0.001,
    weight_decay=1e-3,      # Was 1e-4
)
```

### Fix 3: Update `scripts/train_alphazero.py`

```python
# Line ~180: Reduce exploration
trainer = AlphaZeroTrainer(
    ...
    temp_threshold=20,      # Was 50
)

# Line ~0: Reset training
RESUME_FROM_ITERATION = 0   # Start fresh!
```

### Fix 4: Update `training/alpha_zero/mcts.py`

```python
# Line ~60: Increase MCTS simulations
# In trainer.py when creating MCTS:
mcts = MCTS(
    ...
    num_simulations=800,    # Was 300
)
```

---

## What to Expect After Fixes

### Before Fixes:
```
Iteration 1: loss=3.27, value_loss=1.18, policy_loss=3.10
Iteration 2: loss=2.17, value_loss=1.15, policy_loss=2.00
Iteration 3: loss=1.51, value_loss=1.16, policy_loss=1.33
Iteration 4: loss=1.17, value_loss=1.16, policy_loss=1.00
Iteration 5+: STUCK (loss plateaus)
```

### After Fixes:
```
Iteration 1: loss=3.50, value_loss=1.50, policy_loss=2.00
Iteration 2: loss=2.80, value_loss=1.20, policy_loss=1.60
Iteration 3: loss=2.10, value_loss=0.90, policy_loss=1.20
Iteration 4: loss=1.50, value_loss=0.70, policy_loss=0.80
Iteration 5+: IMPROVING (loss continues to decrease)
```

**Key Difference:** Value loss should **decrease significantly** (1.5 → 0.7), not stay flat (1.18 → 1.16).

---

## Why These Fixes Work

| Fix | Why It Works |
|-----|-------------|
| Increase value loss weight | Value head gets gradient signal to learn |
| Fix draw values | Network learns to distinguish draws from randomness |
| Reduce buffer size | Network trains on fresh data, not stale garbage |
| Reduce Dirichlet noise | Network's policy is heard by MCTS |
| Increase MCTS simulations | Policy targets are less noisy |
| Reduce temp threshold | Less random exploration, more exploitation |
| Stronger regularization | Value head doesn't saturate at ±1 |

---

## Critical: Delete Old Checkpoints

```bash
rm -rf checkpoints/alphazero/checkpoint_iter_*.pth
```

**Why:** The old checkpoints learned to copy randomness. They're corrupted.

---

## Verification Checklist

After applying fixes, check:

- [ ] `value_loss` decreases from 1.5 to <0.5 by iteration 10
- [ ] `policy_loss` decreases from 2.0 to <0.5 by iteration 10
- [ ] `total_loss` decreases consistently (not plateauing)
- [ ] Draw rate decreases from 50% to <30% by iteration 20
- [ ] Win rate increases from 50% to >60% by iteration 20

If any of these don't happen, the fixes didn't work. Check the config values again.

---

## The Root Cause (Technical)

Your system has a **broken learning signal** because:

1. **Value head is starved** (0.15 weight) → can't learn to evaluate positions
2. **Draw values are neutral** (0.0) but network biased toward ±1 → can't learn draws
3. **Replay buffer is stale** (50k with 1.2k new data) → overfitting to old games
4. **Exploration is too aggressive** (25% noise) → network's policy ignored by MCTS

This creates a **negative feedback loop**:
- Network can't learn because it's copying random MCTS
- MCTS is random because network is random
- Network stays random because it's copying random MCTS
- **[STUCK]**

The fixes **break the loop** by:
1. Giving value head gradient signal to learn
2. Making draws learnable (not neutral)
3. Feeding fresh data to the network
4. Letting the network's policy guide MCTS

---

## Timeline to Success

- **Iteration 1-5:** Network learns basic patterns (loss decreases)
- **Iteration 6-15:** Network improves (win rate increases)
- **Iteration 16-30:** Agent becomes competitive (beats random baseline)
- **Iteration 31+:** Agent continues improving (beats previous checkpoints)

If this doesn't happen, something is still wrong. Check the config values.

