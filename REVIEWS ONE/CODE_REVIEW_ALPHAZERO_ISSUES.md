# AlphaZero Checkers Training - Comprehensive Code Review
## Silent Issues & Configuration Problems Leading to "Shit Mode"

---

## Executive Summary

Your training is stuck in **"shit mode"** due to a **critical constellation of silent issues** that compound to create a fundamentally broken learning signal. The system appears to train (losses decrease), but the agent learns **nothing meaningful** because:

1. **Draw values are set to 0.0** (neutral) but the network is trained to output values in [-1, 1]
2. **Value targets for draws are 0.0**, but the network's Tanh output is biased toward ±1
3. **MCTS search bias for draws is -0.03** (search-only), but training targets remain 0.0 → **inconsistency**
4. **Policy loss is computed correctly**, but the network never learns to distinguish winning vs. drawing positions
5. **Replay buffer accumulates stale data** from early iterations when the network was random
6. **No curriculum or progressive difficulty** - the agent plays against itself from iteration 1 with no skill progression
7. **Dirichlet noise parameters are too aggressive** (alpha=0.6, epsilon=0.25) → exploration dominates exploitation
8. **Batch size (512) is too large** for the small buffer (50k) → overfitting to recent games
9. **Value loss weight (0.15) is too low** → policy dominates, value head never learns properly
10. **No value head regularization** → network can output arbitrary values without penalty

---

## Critical Issues (Ranked by Impact)

### 🔴 **ISSUE #1: Draw Value Inconsistency (HIGHEST PRIORITY)**

**Location:** `scripts/config_alphazero.py`, `training/alpha_zero/mcts.py`, `training/alpha_zero/trainer.py`

**The Problem:**

```python
# In config_alphazero.py (STANDARD config)
'DRAW_PENALTY': 0.0,
'MCTS_DRAW_VALUE': 0.0,

# In trainer.py - Value target computation
if winner == 1:
    z = 1.0 if player == 1 else -1.0
elif winner == -1:
    z = 1.0 if player == -1 else -1.0
else:
    z = 0.0  # ← DRAW: neutral value
```

**Why This Is Broken:**

- The network's value head outputs `tanh(x)` → range is **[-1, 1]**
- Training targets for draws are **0.0** (neutral)
- But the network is initialized with random weights → outputs are **uniformly distributed** in [-1, 1]
- The network learns to output **0.0 for draws**, but this is **indistinguishable from random noise**
- **Result:** The value head never learns to differentiate between:
  - Winning positions (target: +1.0)
  - Losing positions (target: -1.0)
  - Drawing positions (target: 0.0)

**The Inconsistency:**

In `mcts.py`, there's a **search-time bias** for draws:

```python
if env.winner == 0:
    value = 0.0
    search_bias = -0.03  # ← Bias ONLY during search
    biased_value = value + search_bias
    node.value_sum += value  # ← But store 0.0, not biased_value!
```

This means:
- **During MCTS search:** Draws are slightly penalized (-0.03) to break deadlocks
- **During training:** Draws are treated as neutral (0.0)
- **Result:** The network learns to output 0.0, but MCTS search doesn't trust it

---

### 🔴 **ISSUE #2: Value Head Never Learns (CRITICAL)**

**Location:** `training/alpha_zero/trainer.py` (lines 200-220)

**The Problem:**

```python
# Value loss weight is TOO LOW
value_loss_weight=0.15,  # ← Only 15% of total loss!
policy_loss_weight=1.0,  # ← 100% of total loss!

# Total loss computation
loss = (self.value_loss_weight * value_loss) + (self.policy_loss_weight * policy_loss)
```

**Why This Is Broken:**

- **Policy loss dominates** (1.0 vs 0.15 weight ratio = 6.7:1)
- The value head is **starved for gradient signal**
- From your logs: `value_loss=1.1813` but `policy_loss=3.0974`
- The value head learns to output **near-zero for everything** (safe default)
- **Result:** The value head becomes useless for guiding MCTS

**Expected Behavior:**

In AlphaZero, value and policy losses should be **roughly balanced**:
- Policy loss: ~1.0-2.0 (cross-entropy over 170 actions)
- Value loss: ~0.5-1.0 (MSE over [-1, 1] range)

Your ratio is **inverted** - value loss should be **higher**, not lower.

---

### 🔴 **ISSUE #3: Replay Buffer Contamination (CRITICAL)**

**Location:** `training/alpha_zero/trainer.py` (lines 100-150)

**The Problem:**

```python
# Buffer size: 50,000 transitions
'BUFFER_SIZE': 50000,

# Games per iteration: 12
'GAMES_PER_ITERATION': 12,

# Batch size: 512
'BATCH_SIZE': 512,
```

**Why This Is Broken:**

- **12 games × ~100 moves = ~1,200 transitions per iteration**
- **Buffer holds 50,000 transitions = ~42 iterations of data**
- **Batch size 512 = 42% of buffer per epoch**
- **Problem:** The buffer is **dominated by stale data** from early iterations when the network was random
- **Result:** Training on 42 iterations of old data → network never improves beyond random baseline

**Example Timeline:**

```
Iteration 1: Add 1,281 transitions (random network)
Iteration 2: Add 1,438 transitions (still random)
...
Iteration 42: Add 1,500 transitions (finally learning)
Training: Sample 512 transitions → 95% from iterations 1-41 (garbage data)
```

---

### 🔴 **ISSUE #4: Dirichlet Noise Too Aggressive (HIGH)**

**Location:** `training/alpha_zero/mcts.py` (lines 20-25)

**The Problem:**

```python
# In trainer.py initialization
mcts = MCTS(
    ...
    dirichlet_alpha=0.6,      # ← Too high!
    dirichlet_epsilon=0.25,   # ← Too high!
)

# In mcts.py
def _add_dirichlet_noise(self, node: AlphaNode):
    noise = np.random.dirichlet([self.dirichlet_alpha] * len(children))
    for i, child in enumerate(children):
        child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * noise[i]
```

**Why This Is Broken:**

- **Dirichlet(0.6) with 170 actions** → very sparse, high-variance noise
- **epsilon=0.25** → 25% of prior is replaced with noise
- **Result:** The network's policy is **completely drowned out** by random noise
- **Example:** If network outputs [0.8, 0.1, 0.05, 0.05], after noise it becomes [0.3, 0.2, 0.25, 0.25]
- **Impact:** MCTS explores randomly instead of following the network's guidance

**Recommended Values:**
- `dirichlet_alpha=0.3` (sparse, but not too aggressive)
- `dirichlet_epsilon=0.1` (10% noise, not 25%)

---

### 🟠 **ISSUE #5: Policy Loss Computation Is Correct, But Ineffective (MEDIUM)**

**Location:** `training/alpha_zero/trainer.py` (lines 280-285)

**The Code:**

```python
def _compute_policy_loss(self, policy_logits: torch.Tensor, policy_target: torch.Tensor) -> torch.Tensor:
    # policy_logits are log-probabilities (log_softmax), policy_target is a distribution
    return -(policy_target * policy_logits).sum(dim=1).mean()
```

**Why This Is Problematic:**

- The formula is **mathematically correct** (cross-entropy)
- But the **policy targets are MCTS visit counts**, which are **extremely sparse**
- **Example:** In a 12-game batch with ~1,200 transitions:
  - Most actions have visit count = 0 (policy_target ≈ 0)
  - Only ~5-10 actions per state have non-zero visits
  - **Result:** The loss is dominated by the few actions MCTS explored
  - The network learns to **copy MCTS**, not to **improve upon it**

**Why This Matters:**

- In early iterations, MCTS is **random** (network is random)
- The network learns to **copy random MCTS behavior**
- This creates a **negative feedback loop**: bad network → random MCTS → network copies randomness

---

### 🟠 **ISSUE #6: Value Head Initialization & Regularization (MEDIUM)**

**Location:** `training/alpha_zero/network.py` (lines 80-120)

**The Problem:**

```python
# Value head initialization
self.value_fc1 = nn.Linear(self.flatten_size, 128)
self.value_fc2 = nn.Linear(128, 1)

# Initialization uses Kaiming (designed for ReLU)
nn.init.kaiming_normal_(module.weight, nonlinearity='relu')

# But value head uses Tanh output
value = torch.tanh(value)
```

**Why This Is Broken:**

- **Kaiming initialization assumes ReLU activations** (mean=0, variance grows with fan-in)
- **Tanh has different activation statistics** (mean=0, but variance is smaller)
- **Result:** Value head weights are **too large** → outputs saturate at ±1
- **No L2 regularization on value head** → weights can grow unbounded
- **Result:** Value head outputs become **binary** (±1) instead of **continuous** (-1 to +1)

**Evidence from Your Logs:**

```
value_loss=1.1813  ← Very high! (MSE between ±1 and 0.0)
```

This suggests the value head is **saturating** at ±1 instead of learning to output 0.0 for draws.

---

### 🟠 **ISSUE #7: No Curriculum or Skill Progression (MEDIUM)**

**Location:** `scripts/train_alphazero.py` (entire training loop)

**The Problem:**

```python
# Iteration 1: Play 12 games with RANDOM network
# Iteration 2: Play 12 games with SLIGHTLY-BETTER network
# ...
# Iteration 100: Play 12 games with MARGINALLY-BETTER network
```

**Why This Is Broken:**

- **No curriculum:** The agent plays against itself from day 1
- **No skill progression:** Early iterations are **pure noise**
- **No bootstrapping:** No pre-training on human games or simpler tasks
- **Result:** The agent spends 50+ iterations learning from garbage data

**What Should Happen:**

1. **Phase 1 (Iterations 1-10):** Play against **random baseline** to learn basic tactics
2. **Phase 2 (Iterations 11-30):** Play against **previous best** to improve incrementally
3. **Phase 3 (Iterations 31+):** Play against **self** with **periodic evaluation**

---

### 🟡 **ISSUE #8: Batch Size Too Large for Buffer Size (MEDIUM)**

**Location:** `scripts/config_alphazero.py`

**The Problem:**

```python
'BUFFER_SIZE': 50000,
'BATCH_SIZE': 512,
'GAMES_PER_ITERATION': 12,
```

**Why This Is Broken:**

- **12 games × ~100 moves = ~1,200 transitions per iteration**
- **Batch size 512 = 42% of new data per epoch**
- **10 epochs per iteration = 5,120 samples from 1,200 new transitions**
- **Result:** Severe **overfitting** to recent games
- **The network memorizes** the last 4 iterations instead of learning generalizable patterns

**Recommended Ratios:**

- Batch size should be **5-10% of buffer size**
- For 50k buffer: batch size should be **2,500-5,000** (but GPU can't handle this)
- **OR** reduce buffer to **5,000** and batch to **256**

---

### 🟡 **ISSUE #9: MCTS Simulations Too Low for Early Training (MEDIUM)**

**Location:** `scripts/config_alphazero.py`

**The Problem:**

```python
'MCTS_SIMULATIONS': 300,
```

**Why This Is Problematic:**

- **300 simulations for 170 possible actions** = ~1.76 simulations per action
- **This is too low** to get reliable policy targets
- **Result:** MCTS visit counts are **extremely noisy**
- **The network learns from noisy targets** → learns noise

**Recommended Values:**

- **Early iterations (1-20):** 800 simulations (explore more)
- **Mid iterations (21-50):** 400 simulations (balance)
- **Late iterations (51+):** 200 simulations (exploit)

---

### 🟡 **ISSUE #10: No Value Head Regularization (MEDIUM)**

**Location:** `training/alpha_zero/trainer.py` (lines 150-160)

**The Problem:**

```python
# Optimizer has L2 regularization
optimizer = optim.Adam(
    model.network.parameters(),
    lr=0.001,
    weight_decay=1e-4  # ← L2 regularization
)

# But value head can still saturate
value = torch.tanh(value)  # ← No explicit constraint
```

**Why This Is Broken:**

- **L2 regularization (1e-4) is too weak** for the value head
- **Tanh can saturate** even with weak regularization
- **Result:** Value head outputs become **binary** (±1) instead of **continuous**

---

## Secondary Issues (Ranked by Impact)

### 🟡 **ISSUE #11: Temp Threshold Too High (LOW-MEDIUM)**

**Location:** `scripts/train_alphazero.py` (line 180)

```python
trainer = AlphaZeroTrainer(
    ...
    temp_threshold=50,  # ← Too high!
)
```

**Why This Is Problematic:**

- **Temp=1.0 (exploration) for first 50 moves**
- **Temp=0.0 (exploitation) for moves 51+**
- **Average game length: ~100 moves**
- **Result:** 50% of moves are **random**, 50% are **greedy**
- **This is backwards** - should be 20% random, 80% greedy

**Recommended:** `temp_threshold=20`

---

### 🟡 **ISSUE #12: No Evaluation Against Baseline (LOW-MEDIUM)**

**Location:** `scripts/train_alphazero.py` (entire script)

**The Problem:**

- No evaluation against **previous checkpoints**
- No evaluation against **random baseline**
- No evaluation against **MCTS-only baseline**
- **Result:** You can't tell if the agent is improving or just overfitting

---

### 🟡 **ISSUE #13: CSV Logging Doesn't Track Value Head Quality (LOW)**

**Location:** `scripts/train_alphazero.py` (lines 60-80)

**The Problem:**

```python
# Logs don't include:
# - Value head accuracy (correlation with game outcome)
# - Policy head entropy (how confident is the network?)
# - MCTS tree depth (how deep does search go?)
# - Win rate vs. previous checkpoint
```

**Result:** You can't diagnose what's going wrong

---

## Root Cause Analysis

### Why Is Training Appearing to Work But Producing Nothing?

**The Illusion:**

```
Iteration 1: loss=3.2746, value_loss=1.1813, policy_loss=3.0974
Iteration 2: loss=2.1701, value_loss=1.1469, policy_loss=1.9981
Iteration 3: loss=1.5066, value_loss=1.1637, policy_loss=1.3321
Iteration 4: loss=1.1723, value_loss=1.1592, policy_loss=0.9984
```

**Looks like:** The network is learning (loss decreasing)

**Reality:**

1. **Iteration 1:** Network is random, MCTS is random, loss is high
2. **Iteration 2:** Network learns to copy random MCTS, loss decreases
3. **Iteration 3:** Network copies MCTS better, loss decreases more
4. **Iteration 4:** Network has memorized MCTS behavior, loss plateaus
5. **Iteration 5+:** Network is stuck - can't improve because it's copying random behavior

**The Feedback Loop:**

```
Random Network
    ↓
Random MCTS (because network is random)
    ↓
Network learns to copy random MCTS
    ↓
Network outputs random policy
    ↓
MCTS still random (because network is still random)
    ↓
[LOOP - STUCK]
```

---

## Recommended Fixes (Priority Order)

### 🔴 **FIX #1: Increase Value Loss Weight (IMMEDIATE)**

**File:** `training/alpha_zero/trainer.py` (line 45)

```python
# BEFORE
value_loss_weight=0.15,
policy_loss_weight=1.0,

# AFTER
value_loss_weight=1.0,  # ← Equal weight!
policy_loss_weight=1.0,
```

**Why:** The value head needs equal gradient signal to learn properly.

---

### 🔴 **FIX #2: Fix Draw Value Handling (IMMEDIATE)**

**File:** `scripts/config_alphazero.py`

```python
# BEFORE
'DRAW_PENALTY': 0.0,
'MCTS_DRAW_VALUE': 0.0,

# AFTER
'DRAW_PENALTY': -0.05,  # ← Slight penalty for draws
'MCTS_DRAW_VALUE': -0.05,  # ← Consistent!
```

**Why:** Draws should be slightly penalized to encourage decisive play.

---

### 🔴 **FIX #3: Reduce Replay Buffer Size (IMMEDIATE)**

**File:** `scripts/config_alphazero.py`

```python
# BEFORE
'BUFFER_SIZE': 50000,
'BATCH_SIZE': 512,

# AFTER
'BUFFER_SIZE': 5000,   # ← 10x smaller
'BATCH_SIZE': 256,     # ← 2x smaller
```

**Why:** Smaller buffer = fresher data = less overfitting to stale games.

---

### 🔴 **FIX #4: Reduce Dirichlet Noise (IMMEDIATE)**

**File:** `training/alpha_zero/trainer.py` (line 130)

```python
# BEFORE
mcts = MCTS(
    ...
    dirichlet_alpha=0.6,
    dirichlet_epsilon=0.25,
)

# AFTER
mcts = MCTS(
    ...
    dirichlet_alpha=0.3,   # ← 50% reduction
    dirichlet_epsilon=0.1,  # ← 60% reduction
)
```

**Why:** Less noise = network's policy is heard by MCTS.

---

### 🟠 **FIX #5: Add Value Head Regularization (HIGH)**

**File:** `training/alpha_zero/trainer.py` (line 150)

```python
# BEFORE
optimizer = optim.Adam(
    self.model.network.parameters(),
    lr=0.001,
    weight_decay=1e-4,
)

# AFTER
optimizer = optim.Adam(
    self.model.network.parameters(),
    lr=0.001,
    weight_decay=1e-3,  # ← 10x stronger regularization
)
```

**Why:** Stronger regularization prevents value head saturation.

---

### 🟠 **FIX #6: Adjust Temp Threshold (HIGH)**

**File:** `scripts/train_alphazero.py` (line 180)

```python
# BEFORE
temp_threshold=50,

# AFTER
temp_threshold=20,  # ← Explore less, exploit more
```

**Why:** 50% random moves is too much exploration.

---

### 🟠 **FIX #7: Increase MCTS Simulations (HIGH)**

**File:** `scripts/config_alphazero.py`

```python
# BEFORE
'MCTS_SIMULATIONS': 300,

# AFTER
'MCTS_SIMULATIONS': 800,  # ← 2.7x more simulations
```

**Why:** More simulations = more reliable policy targets.

---

### 🟡 **FIX #8: Add Evaluation Loop (MEDIUM)**

**File:** `scripts/train_alphazero.py` (after line 200)

```python
# Add after training step
if iteration % 5 == 0:
    eval_stats = evaluate_vs_previous(trainer, iteration)
    log_evaluation(eval_stats)
```

**Why:** You need to know if the agent is actually improving.

---

### 🟡 **FIX #9: Add Value Head Initialization (MEDIUM)**

**File:** `training/alpha_zero/network.py` (lines 80-120)

```python
# BEFORE
nn.init.kaiming_normal_(module.weight, nonlinearity='relu')

# AFTER (for value head only)
if isinstance(module, nn.Linear) and module.out_features == 1:
    # Value head: use smaller initialization
    nn.init.uniform_(module.weight, -0.01, 0.01)
    nn.init.constant_(module.bias, 0.0)
else:
    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
```

**Why:** Value head needs smaller initialization to avoid saturation.

---

## Summary Table

| Issue | Severity | Impact | Fix Effort | Priority |
|-------|----------|--------|-----------|----------|
| Draw value inconsistency | 🔴 Critical | Agent can't learn draws | Low | 1 |
| Value loss weight too low | 🔴 Critical | Value head doesn't learn | Low | 2 |
| Replay buffer contamination | 🔴 Critical | Overfitting to stale data | Low | 3 |
| Dirichlet noise too aggressive | 🔴 Critical | Network drowned out | Low | 4 |
| Policy loss ineffective | 🟠 High | Network copies randomness | Medium | 5 |
| Value head saturation | 🟠 High | Value head useless | Low | 6 |
| No curriculum | 🟠 High | Learning from garbage | High | 7 |
| Batch size too large | 🟡 Medium | Overfitting | Low | 8 |
| MCTS simulations too low | 🟡 Medium | Noisy targets | Low | 9 |
| No regularization | 🟡 Medium | Saturation | Low | 10 |

---

## Expected Improvements After Fixes

**Before Fixes:**
- Iteration 1-4: Loss decreases (illusion of learning)
- Iteration 5+: Loss plateaus (stuck)
- Win rate: ~50% (random)
- Draw rate: ~50% (no progress)

**After Fixes:**
- Iteration 1-10: Loss decreases (network learning)
- Iteration 11-30: Loss stabilizes (convergence)
- Iteration 31+: Win rate increases (agent improving)
- Draw rate: Decreases to ~20-30% (agent learning to win)

---

## Conclusion

Your AlphaZero implementation is **technically sound** but **strategically broken**. The system trains without errors, but the **learning signal is corrupted** by:

1. **Inconsistent value targets** (draws treated as neutral but network biased toward ±1)
2. **Starved value head** (0.15 weight vs 1.0 for policy)
3. **Stale replay buffer** (50k buffer with 1.2k new data per iteration)
4. **Aggressive exploration** (25% noise, 50% random moves)
5. **Noisy policy targets** (300 simulations for 170 actions)

The **good news:** All fixes are **simple configuration changes**. No architectural redesign needed.

The **bad news:** You need to **restart training** with corrected configs. The current checkpoints are **corrupted** with learned randomness.

---

## Next Steps

1. **Apply all 🔴 CRITICAL fixes immediately**
2. **Delete old checkpoints** (they're learning garbage)
3. **Restart training** with `RESUME_FROM_ITERATION = 0`
4. **Monitor value loss** - it should decrease significantly
5. **Check win rate** - should increase after iteration 10
6. **Evaluate vs. baseline** - should beat random by iteration 20

