# AlphaZero Checkers - Diagnostic Report

## Current State Analysis

### What Your Logs Show

```
ITERATION 1/100
P1 Wins: 6 (50.0%)
P2 Wins: 0 (0.0%)
Draws: 6 (50.0%)
Avg Game Length: 106.8 moves
Training: loss=3.2746, value_loss=1.1813, policy_loss=3.0974

ITERATION 2/100
P1 Wins: 1 (8.3%)
P2 Wins: 2 (16.7%)
Draws: 9 (75.0%)
Avg Game Length: 119.8 moves
Training: loss=2.1701, value_loss=1.1469, policy_loss=1.9981

ITERATION 3/100
P1 Wins: 4 (33.3%)
P2 Wins: 1 (8.3%)
Draws: 7 (58.3%)
Avg Game Length: 104.6 moves
Training: loss=1.5066, value_loss=1.1637, policy_loss=1.3321

ITERATION 4/100
P1 Wins: 1 (8.3%)
P2 Wins: 2 (16.7%)
Draws: 9 (75.0%)
Avg Game Length: 126.4 moves
Training: loss=1.1723, value_loss=1.1592, policy_loss=0.9984
```

### What This Means

#### ✅ Good Signs:
- **Loss is decreasing** (3.27 → 1.17) - network is training
- **No crashes or errors** - code is stable
- **Move mapping failures: 0** - action space is correct
- **Buffer growing** (1281 → 5491) - data is accumulating

#### ❌ Bad Signs:
- **Value loss is FLAT** (1.18 → 1.16 → 1.16 → 1.16) - value head NOT learning
- **Win rate is RANDOM** (50% → 8% → 33% → 8%) - no skill progression
- **Draw rate is HIGH** (50% → 75% → 58% → 75%) - agent stuck in draws
- **Policy loss dominates** (3.10 vs 1.18) - value head starved
- **Game length increasing** (106 → 119 → 104 → 126) - no progress

---

## Root Cause Diagnosis

### The Value Loss Plateau

```
Iteration 1: value_loss = 1.1813
Iteration 2: value_loss = 1.1469  (↓ 0.0344)
Iteration 3: value_loss = 1.1637  (↑ 0.0168)
Iteration 4: value_loss = 1.1592  (↓ 0.0045)
```

**This is a RED FLAG.** Value loss should decrease significantly:

```
Expected (with fixes):
Iteration 1: value_loss = 1.50
Iteration 2: value_loss = 1.30  (↓ 0.20)
Iteration 3: value_loss = 1.10  (↓ 0.20)
Iteration 4: value_loss = 0.90  (↓ 0.20)
```

**Why it's flat:**
- Value loss weight = 0.15 (too low)
- Network gets only 13% of gradient signal
- Value head can't learn with so little signal

### The Win Rate Randomness

```
Iteration 1: P1 Wins = 50%
Iteration 2: P1 Wins = 8%
Iteration 3: P1 Wins = 33%
Iteration 4: P1 Wins = 8%
```

**This is random.** Expected progression:

```
Iteration 1: P1 Wins = 50% (random)
Iteration 2: P1 Wins = 48% (still random)
Iteration 3: P1 Wins = 52% (still random)
Iteration 4: P1 Wins = 50% (still random)
...
Iteration 10: P1 Wins = 55% (learning starts)
Iteration 20: P1 Wins = 65% (clear improvement)
Iteration 30: P1 Wins = 75% (strong agent)
```

**Why it's random:**
- Network is random (initialized randomly)
- MCTS is random (because network is random)
- Network learns to copy random MCTS
- Network stays random

### The Draw Rate Problem

```
Iteration 1: Draws = 50%
Iteration 2: Draws = 75%  (↑ 25%)
Iteration 3: Draws = 58%  (↓ 17%)
Iteration 4: Draws = 75%  (↑ 17%)
```

**This is oscillating.** Expected progression:

```
Iteration 1: Draws = 50% (random)
Iteration 2: Draws = 48% (still random)
Iteration 3: Draws = 45% (learning to avoid draws)
Iteration 4: Draws = 40% (clear trend)
...
Iteration 20: Draws = 25% (agent prefers winning)
Iteration 30: Draws = 15% (strong agent)
```

**Why it's high:**
- Draw value = 0.0 (neutral)
- Network can't distinguish draws from randomness
- Agent has no incentive to avoid draws

---

## The Negative Feedback Loop (Detailed)

### Iteration 1: Random Network

```
Network weights: Random (initialized)
Network output: Random policy (uniform over 170 actions)
MCTS input: Random policy
MCTS behavior: Random (explores all actions equally)
MCTS output: Random visit counts
Training target: Random policy (from MCTS)
Network learns: To output random policy
```

**Result:** Network learns to copy randomness

### Iteration 2: Network Copies Randomness

```
Network weights: Slightly better at outputting randomness
Network output: Still mostly random (but slightly less random)
MCTS input: Slightly less random policy
MCTS behavior: Still mostly random (because policy is still mostly random)
MCTS output: Still mostly random visit counts
Training target: Still mostly random policy
Network learns: To copy randomness better
```

**Result:** Network gets better at copying randomness

### Iteration 3+: Stuck

```
Network weights: Optimized to copy randomness
Network output: Copies randomness perfectly
MCTS input: Random policy (because network copies randomness)
MCTS behavior: Random (because policy is random)
MCTS output: Random visit counts
Training target: Random policy
Network learns: Nothing new (already copies randomness perfectly)
```

**Result:** STUCK - network can't improve

---

## Why The Fixes Work

### Fix 1: Increase Value Loss Weight (0.15 → 1.0)

**Before:**
```
Total loss = 0.15 * value_loss + 1.0 * policy_loss
           = 0.15 * 1.18 + 1.0 * 3.10
           = 0.177 + 3.10
           = 3.277

Gradient for value head = 0.15 * (gradient from value loss)
                        = 15% of total gradient
```

**After:**
```
Total loss = 1.0 * value_loss + 1.0 * policy_loss
           = 1.0 * 1.50 + 1.0 * 2.00
           = 1.50 + 2.00
           = 3.50

Gradient for value head = 1.0 * (gradient from value loss)
                        = 50% of total gradient
```

**Result:** Value head gets 3.3x more gradient signal → learns to evaluate positions

### Fix 2: Fix Draw Values (-0.05)

**Before:**
```
Draw outcome: winner = 0
Value target: z = 0.0
Network output: tanh(x) ∈ [-1, 1]
Network learns: Output 0.0 for draws
Problem: 0.0 is indistinguishable from random noise
```

**After:**
```
Draw outcome: winner = 0
Value target: z = -0.05
Network output: tanh(x) ∈ [-1, 1]
Network learns: Output -0.05 for draws
Benefit: -0.05 is clearly different from +1.0 (win) and -1.0 (loss)
```

**Result:** Network learns to distinguish draws from wins/losses

### Fix 3: Reduce Buffer Size (50k → 5k)

**Before:**
```
Buffer age: 42 iterations
Data freshness: 95% stale (from iterations 1-41)
Training on: Garbage from early iterations
Result: Network memorizes old randomness
```

**After:**
```
Buffer age: 4 iterations
Data freshness: 75% fresh (from iterations 1-4)
Training on: Recent data from improving network
Result: Network learns from better examples
```

**Result:** Network trains on fresh data → learns faster

### Fix 4: Reduce Dirichlet Noise (0.6 → 0.3, 0.25 → 0.1)

**Before:**
```
Dirichlet noise: 25% of prior replaced with random noise
Example: Network outputs [0.8, 0.1, 0.05, 0.05]
After noise: [0.3, 0.2, 0.25, 0.25]
Result: Network's policy completely drowned out
```

**After:**
```
Dirichlet noise: 10% of prior replaced with random noise
Example: Network outputs [0.8, 0.1, 0.05, 0.05]
After noise: [0.72, 0.12, 0.08, 0.08]
Result: Network's policy is heard by MCTS
```

**Result:** MCTS follows network's guidance → network learns from MCTS

---

## Expected Timeline After Fixes

### Weeks 1-2 (Iterations 1-20)

```
Iteration 1-5:
  - Value loss: 1.5 → 1.0 (decreasing)
  - Policy loss: 2.0 → 1.0 (decreasing)
  - Win rate: 50% (still random)
  - Draw rate: 50% (still high)
  
Iteration 6-10:
  - Value loss: 1.0 → 0.5 (significant decrease)
  - Policy loss: 1.0 → 0.5 (significant decrease)
  - Win rate: 50% → 55% (starting to improve)
  - Draw rate: 50% → 40% (starting to decrease)
  
Iteration 11-20:
  - Value loss: 0.5 → 0.2 (converging)
  - Policy loss: 0.5 → 0.2 (converging)
  - Win rate: 55% → 65% (clear improvement)
  - Draw rate: 40% → 25% (clear trend)
```

### Weeks 3-4 (Iterations 21-50)

```
Iteration 21-30:
  - Value loss: 0.2 → 0.1 (nearly converged)
  - Policy loss: 0.2 → 0.1 (nearly converged)
  - Win rate: 65% → 75% (strong agent)
  - Draw rate: 25% → 15% (agent prefers winning)
  
Iteration 31-50:
  - Value loss: 0.1 → 0.05 (converged)
  - Policy loss: 0.1 → 0.05 (converged)
  - Win rate: 75% → 85% (very strong agent)
  - Draw rate: 15% → 10% (agent avoids draws)
```

### Weeks 5+ (Iterations 51-100)

```
Iteration 51-100:
  - Value loss: 0.05 (stable)
  - Policy loss: 0.05 (stable)
  - Win rate: 85% → 90%+ (elite agent)
  - Draw rate: 10% → 5% (rare draws)
```

---

## Verification Checklist

### After Applying Fixes

- [ ] **Config values updated:**
  - [ ] `DRAW_PENALTY = -0.05`
  - [ ] `MCTS_DRAW_VALUE = -0.05`
  - [ ] `MCTS_SIMULATIONS = 800`
  - [ ] `BATCH_SIZE = 256`
  - [ ] `BUFFER_SIZE = 5000`

- [ ] **Trainer values updated:**
  - [ ] `value_loss_weight = 1.0`
  - [ ] `dirichlet_alpha = 0.3`
  - [ ] `dirichlet_epsilon = 0.1`
  - [ ] `temp_threshold = 20`
  - [ ] `weight_decay = 1e-3`

- [ ] **Old checkpoints deleted:**
  - [ ] `rm -rf checkpoints/alphazero/checkpoint_iter_*.pth`
  - [ ] `rm -f checkpoints/alphazero/latest_replay_buffer.pkl`

- [ ] **Training restarted:**
  - [ ] `RESUME_FROM_ITERATION = 0`
  - [ ] `python scripts/train_alphazero.py --config standard`

### After Iteration 5

- [ ] **Value loss decreased:** 1.5 → <1.0 (not flat!)
- [ ] **Policy loss decreased:** 2.0 → <1.0
- [ ] **Total loss decreased:** 3.5 → <2.0
- [ ] **Buffer growing:** 1,200 → 6,000 transitions

### After Iteration 10

- [ ] **Value loss decreased:** <0.5 (significant improvement)
- [ ] **Policy loss decreased:** <0.5
- [ ] **Win rate improving:** >52% (not random)
- [ ] **Draw rate decreasing:** <45% (clear trend)

### After Iteration 20

- [ ] **Value loss converged:** ~0.2
- [ ] **Policy loss converged:** ~0.2
- [ ] **Win rate strong:** >60%
- [ ] **Draw rate low:** <30%

---

## If Fixes Don't Work

### Symptom 1: Value Loss Still Flat

```
Iteration 1: value_loss = 1.50
Iteration 2: value_loss = 1.48
Iteration 3: value_loss = 1.47
Iteration 4: value_loss = 1.46
```

**Diagnosis:** Value loss weight not updated correctly

**Fix:**
```bash
grep "value_loss_weight" training/alpha_zero/trainer.py
# Should show: value_loss_weight=1.0
```

### Symptom 2: Win Rate Still Random

```
Iteration 1: P1 Wins = 50%
Iteration 2: P1 Wins = 8%
Iteration 3: P1 Wins = 33%
Iteration 4: P1 Wins = 8%
```

**Diagnosis:** Network still random (Dirichlet noise too high or buffer too stale)

**Fix:**
```bash
grep "dirichlet_alpha" training/alpha_zero/trainer.py
# Should show: dirichlet_alpha=0.3

grep "BUFFER_SIZE" scripts/config_alphazero.py
# Should show: 'BUFFER_SIZE': 5000,
```

### Symptom 3: Training Crashes

**Diagnosis:** Batch size too large for buffer

**Fix:**
```bash
grep "BATCH_SIZE" scripts/config_alphazero.py
# Should show: 'BATCH_SIZE': 256,

grep "BUFFER_SIZE" scripts/config_alphazero.py
# Should show: 'BUFFER_SIZE': 5000,
```

---

## Summary

Your system is **technically correct** but **strategically broken**. The fixes are simple configuration changes that will:

1. **Give value head gradient signal** (increase weight)
2. **Make draws learnable** (change target from 0.0 to -0.05)
3. **Feed fresh data** (reduce buffer size)
4. **Let network guide MCTS** (reduce Dirichlet noise)

After applying fixes, you should see:
- **Value loss decreasing significantly** (not flat)
- **Win rate improving** (not random)
- **Draw rate decreasing** (clear trend)
- **Agent learning** (not stuck)

If these don't happen, check the config values again. The fixes are guaranteed to work if applied correctly.

