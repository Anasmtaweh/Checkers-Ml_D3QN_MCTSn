# Checkers ML: Gen 10 Titan Training Documentation

**Author**: ML Engineering Team  
**Date**: December 21, 2025  
**Project**: D3QN Deep Reinforcement Learning for Checkers  
**Phase**: Gen 9 → Gen 10 Transition (Endgame Curriculum)

---

## Executive Summary

This document details the strategic modifications to transition from **Gen 9 Titan** (strong opening, weak endgame) to **Gen 10 Titan** (balanced play across all phases). The core issue identified was **late-game collapse** caused by temporal credit assignment failure and endgame position undersampling in the replay buffer.

**Solution**: Non-invasive endgame curriculum injection with stabilized learning rate controls.

---

## Problem Statement

### Gen 9 Titan Performance Analysis

From resume training at checkpoint 4500 (Episode 4510-4950):

| Metric | Observation | Root Cause |
|--------|-------------|-----------|
| **Early Game** | ✅ 40-50% WR vs pool | Clear piece advantages, many legal moves |
| **Mid-to-Endgame** | ⚠️ 20-35% WR vs self | Fewer pieces = harder credit assignment |
| **Endgame (6-8 pieces)** | ❌ <20% WR | Underrepresented in training data (~5-10%) |
| **Q-Value Health** | Exploded 40→45 | Auto-brake too aggressive (threshold=35.0) |
| **Learning Rate** | Collapsed 5e-7→1e-8 | Premature braking prevented adaptation |

### Why Late-Game Fails

1. **Temporal Credit Assignment**: By endgame, moves made 20+ steps earlier now determine win/loss. The agent struggles to correlate early king positioning with endgame outcomes.

2. **Replay Buffer Bias**: Most games end at move 50-100 (midgame). Endgame positions represent only 5-10% of buffer samples. Gradient updates underrepresent endgame optimal policies.

3. **Exploration Collapse**: With high epsilon decay (0.3→0.05 over 5000 eps), by ep 4500 epsilon is near 0.05. Agent exploits suboptimal endgame policies instead of exploring.

4. **King Movement Patterns**: Endgames dominated by king movement (double pieces). Gen 9 trained mainly on regular pieces; king behavior underdeveloped.

---

## Solution Architecture

### 1. Non-Invasive Endgame Injection

**Philosophy**: Inject synthetic endgame positions directly into `env.board.board` without modifying core environment code.

**Mechanism**:
```
inject_random_endgame(env, min_pieces=6, max_pieces=10)
├─ Generate blank 8x8 board
├─ Select valid dark squares (checkers board)
├─ Place 6-10 pieces randomly (balanced sides, 50% kings)
├─ Validate legal moves exist
├─ Return state to continue training
```

**Why Non-Invasive**:
- ✅ No changes to `CheckersEnv` class
- ✅ No changes to `Board` class
- ✅ Modular and reversible
- ✅ Can toggle on/off via activation probability (40%)

---


### 2. Auto-Brake System Improvements

#### Previous Problem (Checkpoint 4500-4950)

| Episode | Max Q | LR Before | LR After | Reason |
|--------|-------|-----------|----------|--------|
| 4540 | 40.0 | 5e-7 | 2.5e-7 | Brake triggered at 35.0 threshold |
| 4600 | 40.6 | 2.5e-7 | 1.25e-7 | Brake triggered again |
| 4800 | 41.7 | 3.12e-8 | 1e-8 | Hit minimum floor |
| 4950 | 44.5 | 1e-8 | 1e-8 | **Stuck at floor, no learning** |

**Root Cause**: Auto-brake threshold (35.0) was calibrated for early training. Gen 9 naturally operates at Q~40-45, causing false positives.

#### Solution: Threshold & Floor Adjustment

```python
# OLD (Lines 386-393)
if max_q > 35.0:  # TOO AGGRESSIVE
    new_lr = max(old_lr * 0.5, 1e-8)  # TOO HIGH FLOOR

# NEW (Lines 386-393)
if max_q > 50.0:  # RATIONAL THRESHOLD
    new_lr = max(old_lr * 0.5, 2e-8)  # LOWER FLOOR
```

**Rationale**:
- **35.0→50.0**: Gen 9 operates at 40-47. Threshold should be 3x higher than normal peak.
- **1e-8→2e-8**: Lowers floor but maintains atomic-level precision. Prevents permanent learning shutdown.
- **Emergency stop unchanged at 60.0**: True explosion detection remains conservative.

---


### 3. Extended Game Length for Endgames

```python
# OLD
MAX_MOVES_PER_GAME = 500

# NEW
MAX_MOVES_PER_GAME = 600
```

**Why**: Endgame positions with 6-8 pieces often require 100+ moves to resolve (king maneuvers, draws). 500 moves insufficient; games forced to draw artificially.

---


### 4. Custom Reward Structure (Endgame-Aware)

```python
# Replaces flat environment reward
if done and winner == agent_side:
    custom_reward = 1.0  # Win = +1
elif done and winner == -agent_side:
    custom_reward = -1.0  # Loss = -1
elif done:
    custom_reward = 0.0  # Draw = 0
else:
    # Shaping rewards for non-terminal states
    if reward > 20.0:
        custom_reward = 0.01  # Major advantage (capture many pieces)
    elif reward > 8.0:
        custom_reward = 0.001  # Moderate advantage (single capture)
    else:
        custom_reward = -0.0001  # Slow gradual discount (no progress)
```

**Benefits**:
- ✅ Terminal wins/losses weighted at ±1.0
- ✅ Intermediate progress tracked but discounted
- ✅ Prevents reward scaling issues causing Q-explosion
- ✅ Endgame positions reached with clear target signals

---


### 5. Endgame-Specific Opponent Selection

```python
# NEW LOGIC
if is_endgame_mode:
    # Face stronger opponents in endgame training
    p_rand, p_pool, p_self = 0.10, 0.60, 0.30
else:
    # Normal distribution
    p_rand, p_pool, p_self = get_opponent_probabilities(episode)
```

**Why**: 
- Endgame positions are rare in random play
- Stronger pool opponents provide endgame expertise
- Self-play at 6-8 pieces is weaker teacher (agent hasn't learned endgame yet)
- 60% pool + 30% self = 90% structured opponents

---


### 6. Fixed Import Paths

```python
# OLD (BROKEN)
from training.common.action_manager import ActionManager
from training.d3qn.model import D3QNModel

# NEW (CORRECTED)
from common.action_manager import ActionManager
from d3qn_legacy.d3qn.model import D3QNModel
```

**Reason**: Repository structure doesn't have `training/` root folder. Correct paths:
- `common/` lives at project root
- `d3qn_legacy/d3qn/` contains model & trainer

---


## Implementation Details

### Endgame Injection Algorithm

```
inject_random_endgame(env, min_pieces=6, max_pieces=10)

STEP 1: Initialize blank 8x8 board (zeros)
STEP 2: Identify 32 valid dark squares (checkers only plays on dark)
STEP 3: Shuffle valid squares randomly
STEP 4: Place N pieces (6-10 random) with constraints:
        - At least 2 P1 pieces (positive values)
        - At least 2 P2 pieces (negative values)
        - 50% probability of king (value = 2 or -2)
        - Back row pieces automatically promoted to kings
STEP 5: Validate environment can calculate legal moves
        - If no legal moves → fallback to normal reset
        - If exception → fallback to normal reset
STEP 6: Return injected state to continue training

Key Safety Features:
✓ Minimum piece constraints prevent instant-win positions
✓ King probability matches natural endgame composition
✓ Automatic fallback prevents training crashes
✓ Non-destructive (doesn't modify checkpoint system)
```

---


## Configuration Changes Summary

### Learning Rate Schedule

```python
def get_learning_rate(episode):
    if episode < 1000:
        return 2e-6  # Aggressive early learning
    elif episode < 3000:
        return 1e-6  # Standard mid-phase
    elif episode < 6000:
        return 5e-7  # Safe zone (ENDGAME PHASE)
    else:
        return 2e-7  # Ultra-fine tuning (100K+ episodes)
```

**Endgame Phase (Episodes 5000-6000)**:
- Base LR = 5e-7 (safe, slow adjustments)
- Auto-brake can cut to 2.5e-8 minimum
- 40% of games are endgame-injected
- Opponent pool: 60% (strong endgame teachers)

---


### Monitoring & Logging

```python
Heartbeat every 100 episodes:
  💓 Entering Episode X
    [Q-Health] Max: {max_q:.2f} | Avg: {avg_q:.2f} | LR: {current_lr:.2e}
    [Memory] CUDA cache cleared

Per-game logging (every 10 episodes):
  Ep XXXXX | [ENDGAME/NORMAL] Red (P1) | Vs Pool: gen9_titan.pth | WIN
  WR[P1]: 43% | WR[P2]: 37% | vsRand: 85%

CSV Log (iron_league_nuclear.csv):
  Episode | Side | Opponent | Result | Reward | P1_WR | P2_WR | vsRand_WR | Loss | Epsilon | LR | MaxQ | AvgQ
```

---


## Expected Outcomes: Gen 9 → Gen 10

### Training Phases

| Phase | Episodes | Focus | Expected Outcome |
|-------|----------|-------|------------------|
| **Phase A** | 5000-5200 | Endgame injection active, buffer filling | Win rates stabilize, Q-values normalize |
| **Phase B** | 5200-5500 | Opponent pool strengthens endgame | P1 WR vs pool >50%, endgame draws reduce |
| **Phase C** | 5500-6000 | Fine-tuning, king maneuver optimization | All win rates converge >65% (endgame specialist) |
| **Phase D** | 6000+ | Evaluation & promotion to Gen 10 | Save as gen10_titan.pth if criteria met |

### Success Criteria for Gen 10 Promotion

```
MUST HAVE (All Required):
✅ Win rate vs Random > 85%
✅ Win rate vs Gen 8 Titan > 60%
✅ Win rate vs Gen 9 Champion > 50%
✅ Max Q-value < 50.0 (stable, not exploding)
✅ Training loss < 0.5 (converged)

NICE TO HAVE (2+ of 3):
⭐ Endgame win rate (6-8 pieces) > 70%
⭐ Drawing capability (draw % > 20% vs strong opponents)
⭐ Opening phase loss < 1% (preserved Gen 9 strength)
```

---


## Architectural Changes Summary

### What Changed
1. ✅ **Import paths**: Fixed to match actual repo structure
2. ✅ **Auto-brake system**: Threshold 35→50, floor 1e-8→2e-8
3. ✅ **Game length**: 500→600 moves (endgame support)
4. ✅ **Reward shaping**: Custom rewards replacing raw env rewards
5. ✅ **Endgame injection**: 40% of games start from 6-10 piece positions
6. ✅ **Opponent selection**: Endgame games face 60% pool, 30% self, 10% random
7. ✅ **Logging**: Added [ENDGAME/NORMAL] mode indicator

### What Stayed the Same
1. ✅ **Core environment** (`CheckersEnv`) - No modifications
2. ✅ **D3QN model architecture** - No changes
3. ✅ **Replay buffer** - Structure unchanged (just different data)
4. ✅ **Training loop** - Same optimizer, same GAMMA (0.99)
5. ✅ **Checkpoint system** - Compatible with Gen 9 resumption

---


## Why This Works

### Addressing Root Causes

| Problem | Root Cause | Solution | Why It Works |
|-----------|-----------|----------|-------------|
| Late-game collapse | Undersampled endgame positions | Inject endgame 40% of time | Buffer now 40% endgame transitions, improves credit assignment |
| Q-value explosion | False auto-brake triggers | Raise threshold 35→50 | Gen 9's natural Q-range no longer triggers unnecessary braking |
| Learning stagnation | LR hit minimum (1e-8) | Lower floor to 2e-8 | Allows continued fine-tuning even after brake activations |
| King underdevelopment | Few kings in training data | 50% king probability in injection | Endgame positions naturally weighted toward kings |
| Weak endgame opponents | Self-play hasn't learned endgame | Use pool opponents (60%) in endgame | Gen 8 Titan provides endgame strategy to learn from |
| Reward scaling issues | Raw environment rewards → Q-explosion | Custom reward shaping | Terminal states weighted 1.0, prevents cascading magnitude growth |

---


## Testing & Validation

### Pre-Training Checklist

Before running 1000+ episodes, validate:

```bash
# 1. Test endgame injection (10 trials)
python -c "
import sys; sys.path.insert(0, '..')
from checkers_env.env import CheckersEnv
import numpy as np

env = CheckersEnv()
for i in range(10):
    state = inject_random_endgame(env, 6, 10)
    p1_pieces = np.sum(state > 0)
    p2_pieces = np.sum(state < 0)
    print(f'Trial {i+1}: P1={p1_pieces}, P2={p2_pieces}, Legal moves: {len(env.get_legal_moves())}')
    assert p1_pieces >= 2 and p2_pieces >= 2, f'Invalid piece count at trial {i+1}'
print('✅ Endgame injection validated')
"

# 2. Verify import paths
python -c "
import sys; sys.path.insert(0, '..')
from common.action_manager import ActionManager
from common.board_encoder import CheckersBoardEncoder
from common.buffer import ReplayBuffer
from d3qn_legacy.d3qn.model import D3QNModel
from d3qn_legacy.d3qn.trainer import D3QNTrainer
print('✅ All imports successful')
"

# 3. Test checkpoint loading
python -c "
import torch
import sys; sys.path.insert(0, '..')
from d3qn_legacy.d3qn.model import D3QNModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = D3QNModel(170, device)
checkpoint = torch.load('checkpoints_iron_league_v3/iron_nuclear_4500.pth', map_location=device)
model.online.load_state_dict(checkpoint['model_online'])
print(f'✅ Checkpoint loaded: Episode {checkpoint["episode"]}')
"
```

---


## Future Enhancements (Post-Gen 10)

1. **Dynamic Gamma**: Increase to 0.995+ for episodes 6000+ (far-sighted endgame planning)
2. **Curriculum Scheduling**: Vary endgame injection rate (lower in late stages, higher early)
3. **MCTS Integration**: Hybrid MCTS+D3QN for piece-count < 8 (future phase)
4. **Prioritized Replay**: Weight endgame transitions higher in sampling
5. **Multi-Agent Tournament**: Evolve multiple agents in parallel (Gen 10a, 10b, 10c variants)

---


## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'training'"

**Solution**: Update import paths:
```python
# Change from:
from training.common.action_manager import ActionManager

# To:
from common.action_manager import ActionManager
```

---


### Issue: "Auto-brake keeps triggering, LR stuck at 2e-8"

**Diagnosis**: Max Q still exceeding 50.0 frequently

**Solutions**:
1. Increase threshold to 55.0
2. Use lower base LR in schedule (5e-8 instead of 5e-7)
3. Reduce batch size from 128 to 64 (less aggressive gradient steps)
4. Increase tau from 0.001 to 0.01 (slower target network updates)

---


### Issue: "Endgame injection creates invalid positions (no legal moves)"

**Diagnosis**: `inject_random_endgame` falling back to normal reset too often

**Solution**: Ensure minimum piece separation:
```python
# Add piece placement distance constraint
valid_squares_shuffled = [...] 
for i in range(count):
    r, c = valid_squares_shuffled[i]
    # Skip squares adjacent to existing pieces
    if any(new_board[nr][nc] != 0 for nr in [r-1,r,r+1] for nc in [c-1,c,c+1]):
        continue
    # ... place piece
```

---


### Issue: "CUDA out of memory during endgame training"

**Solution**: Clear cache more frequently:
```python
# Change from every 100 episodes to every 50:
if episode % 50 == 0:  # Changed from 100
    if device == "cuda":
        torch.cuda.empty_cache()
```

---



## Conclusion

The transition from Gen 9 to Gen 10 addresses a specific weakness: **late-game decision-making under king-piece dominance**. By injecting endgame positions directly into training (40% of episodes) while maintaining all original architecture, we achieve curriculum learning **without invasive environmental changes**.

The stabilized auto-brake system (threshold 50.0) prevents learning collapse while allowing continued gradient updates. Custom reward shaping ensures Q-values remain in healthy ranges despite extended gameplay.

**Expected Timeline**:
- Episodes 5000-5500: Endgame buffer accumulation phase
- Episodes 5500-6000: Endgame specialization phase  
- Episode 6000: Promotion decision (Gen 10 Titan qualification)

**Success Metric**: Gen 10 Titan should achieve **>70% win rate in synthetic endgame positions (6-8 pieces)** while maintaining **>85% vs random** and **>50% vs Gen 9 Champion**.

---


**Document Version**: 1.0  
**Last Updated**: December 21, 2025  
**Status**: Ready for Implementation
