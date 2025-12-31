# Exact Code Changes - Draw Inflation Fixes

## File 1: scripts/config_alphazero.py

### Change: Added phased_curriculum configuration

**Location**: After 'standard' config, before 'production' config

**Added Code**:
```python
# Phased curriculum training (FIXES DRAW INFLATION)
'phased_curriculum': {
    'NUM_ITERATIONS': 100,
    'GAMES_PER_ITERATION': 12,
    'TRAIN_EPOCHS': 10,
    'BATCH_SIZE': 256,
    'BUFFER_SIZE': 5000,
    'description': 'Phased curriculum to reduce draw inflation',
    # Phase-based parameters (iteration ranges)
    'phases': [
        {
            'name': 'Phase A: Early Exploration (Iter 1-10)',
            'iter_start': 1,
            'iter_end': 10,
            'MCTS_SIMULATIONS': 400,
            'DIRICHLET_EPSILON': 0.15,
            'TEMP_THRESHOLD': 15,
            'NO_PROGRESS_PLIES': 60,
            'ENV_MAX_MOVES': 180,
            'DRAW_PENALTY': -0.05,
            'MCTS_DRAW_VALUE': -0.06,
            'MCTS_SEARCH_DRAW_BIAS': -0.06,
        },
        {
            'name': 'Phase B: Balanced Growth (Iter 11-30)',
            'iter_start': 11,
            'iter_end': 30,
            'MCTS_SIMULATIONS': 600,
            'DIRICHLET_EPSILON': 0.10,
            'TEMP_THRESHOLD': 20,
            'NO_PROGRESS_PLIES': 70,
            'ENV_MAX_MOVES': 190,
            'DRAW_PENALTY': -0.05,
            'MCTS_DRAW_VALUE': -0.05,
            'MCTS_SEARCH_DRAW_BIAS': -0.05,
        },
        {
            'name': 'Phase C: Full Strength (Iter 31+)',
            'iter_start': 31,
            'iter_end': 1000,
            'MCTS_SIMULATIONS': 800,
            'DIRICHLET_EPSILON': 0.10,
            'TEMP_THRESHOLD': 20,
            'NO_PROGRESS_PLIES': 80,
            'ENV_MAX_MOVES': 200,
            'DRAW_PENALTY': -0.05,
            'MCTS_DRAW_VALUE': -0.05,
            'MCTS_SEARCH_DRAW_BIAS': -0.03,
        },
    ]
},
```

---

## File 2: training/alpha_zero/mcts.py

### Change 1: Added search_draw_bias parameter to __init__

**Location**: Line 32-40 (in __init__ method)

**Before**:
```python
def __init__(self, model, action_manager, encoder, c_puct: float = 1.5, 
             num_simulations: int = 400, device: str = "cpu",
             dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25,
             draw_value: float = -0.1):
```

**After**:
```python
def __init__(self, model, action_manager, encoder, c_puct: float = 1.5, 
             num_simulations: int = 400, device: str = "cpu",
             dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25,
             draw_value: float = -0.1, search_draw_bias: float = -0.03):
```

### Change 2: Store search_draw_bias in __init__

**Location**: Line 41-48 (in __init__ method)

**Added**:
```python
self.search_draw_bias = float(search_draw_bias)
```

### Change 3: Use configurable search_draw_bias in _search

**Location**: Line 77-88 (in _search method)

**Before**:
```python
if env.winner == 0:
    value = 0.0
    
    # DRAW AVERSION: Small bias during search to break deadlocks
    # This makes the agent prefer a non-drawing line if all else is equal
    # We do NOT use this for training targets (which remain 0.0)
    search_bias = -0.03 
    biased_value = value + search_bias
    
    node.value_sum += value
    node.visits += 1
    return biased_value
```

**After**:
```python
if env.winner == 0:
    value = 0.0
    
    # DRAW AVERSION: Configurable bias during search to break deadlocks
    # This makes the agent prefer a non-drawing line if all else is equal
    # We do NOT use this for training targets (which remain 0.0)
    biased_value = value + self.search_draw_bias
    
    node.value_sum += value
    node.visits += 1
    return biased_value
```

---

## File 3: training/alpha_zero/trainer.py

### Change 1: Added dirichlet_epsilon parameter to __init__

**Location**: Line 32-48 (in __init__ method signature)

**Added**:
```python
dirichlet_epsilon: float = 0.1,
```

### Change 2: Added policy entropy logging in train_step

**Location**: Line 330-340 (in train_step method)

**Before**:
```python
with torch.no_grad():
    tgt_sum = policy_targets.sum(dim=1).mean().item()
    pred_sum = torch.exp(policy_logits).sum(dim=1).mean().item()
print(f"  policy_targets_sum={tgt_sum:.6f}, exp(policy_logits)_sum={pred_sum:.6f}")
```

**After**:
```python
with torch.no_grad():
    tgt_sum = policy_targets.sum(dim=1).mean().item()
    pred_sum = torch.exp(policy_logits).sum(dim=1).mean().item()
    # Compute policy entropy for diagnostics
    probs = torch.exp(policy_logits)
    entropy = -(probs * policy_logits).sum(dim=1).mean().item()
print(f"  policy_targets_sum={tgt_sum:.6f}, exp(policy_logits)_sum={pred_sum:.6f}, avg_policy_entropy={entropy:.4f}")
```

---

## File 4: scripts/train_alphazero.py

### Change 1: Added base config extraction

**Location**: Line 95-103 (after loading CFG)

**Added**:
```python
# Get base config (non-phased values)
env_max_moves = CFG.get('ENV_MAX_MOVES', 200)
no_progress_plies = CFG.get('NO_PROGRESS_PLIES', 80)
draw_penalty = CFG.get('DRAW_PENALTY', -0.1)
mcts_draw_value = CFG.get('MCTS_DRAW_VALUE', draw_penalty)
dirichlet_epsilon = CFG.get('DIRICHLET_EPSILON', 0.1)
temp_threshold = CFG.get('TEMP_THRESHOLD', 20)
mcts_simulations = CFG.get('MCTS_SIMULATIONS', 400)
search_draw_bias = CFG.get('MCTS_SEARCH_DRAW_BIAS', -0.03)
```

### Change 2: Added phase-based parameter application in training loop

**Location**: Line 180-210 (in training loop, after iteration header)

**Added**:
```python
# Apply phase-specific parameters if using phased curriculum
if 'phases' in CFG:
    phase_cfg = None
    for phase in CFG['phases']:
        if phase['iter_start'] <= iteration <= phase['iter_end']:
            phase_cfg = phase
            break
    
    if phase_cfg:
        print(f"\n📋 {phase_cfg['name']}")
        # Update MCTS parameters
        mcts.num_simulations = phase_cfg.get('MCTS_SIMULATIONS', mcts.num_simulations)
        mcts.dirichlet_epsilon = phase_cfg.get('DIRICHLET_EPSILON', mcts.dirichlet_epsilon)
        mcts.search_draw_bias = phase_cfg.get('MCTS_SEARCH_DRAW_BIAS', mcts.search_draw_bias)
        
        # Update trainer parameters
        trainer.temp_threshold = phase_cfg.get('TEMP_THRESHOLD', trainer.temp_threshold)
        trainer.env_max_moves = phase_cfg.get('ENV_MAX_MOVES', trainer.env_max_moves)
        trainer.no_progress_plies = phase_cfg.get('NO_PROGRESS_PLIES', trainer.no_progress_plies)
        trainer.draw_penalty = phase_cfg.get('DRAW_PENALTY', trainer.draw_penalty)
        mcts.draw_value = phase_cfg.get('MCTS_DRAW_VALUE', mcts.draw_value)
        
        print(f"  MCTS sims: {mcts.num_simulations}, Dirichlet ε: {mcts.dirichlet_epsilon}, Search bias: {mcts.search_draw_bias}")
        print(f"  Temp threshold: {trainer.temp_threshold}, Max moves: {trainer.env_max_moves}, No-progress: {trainer.no_progress_plies}")
```

---

## Summary of Changes

### Total Files Modified: 4

1. **scripts/config_alphazero.py**
   - Added 1 new configuration (phased_curriculum)
   - ~60 lines added

2. **training/alpha_zero/mcts.py**
   - Added 1 parameter to __init__
   - Modified 1 method (_search)
   - ~5 lines added/modified

3. **training/alpha_zero/trainer.py**
   - Added 1 parameter to __init__
   - Modified 1 method (train_step)
   - ~5 lines added/modified

4. **scripts/train_alphazero.py**
   - Added base config extraction
   - Added phase application logic
   - ~35 lines added

### Total Lines Added: ~105 lines

### Backward Compatibility: ✅ Fully maintained
- All new parameters have defaults
- Existing configs still work
- No breaking changes

### Testing: ✅ Ready
- Can run with `--config phased_curriculum`
- Can run with `--config standard` (unchanged)
- Can resume from any iteration

---

## Verification Commands

```bash
# Verify config has phased_curriculum
grep -c "phased_curriculum" scripts/config_alphazero.py
# Expected output: 1

# Verify MCTS has search_draw_bias
grep -c "search_draw_bias" training/alpha_zero/mcts.py
# Expected output: 3 (parameter, assignment, usage)

# Verify trainer has entropy logging
grep -c "avg_policy_entropy" training/alpha_zero/trainer.py
# Expected output: 1

# Verify training script has phase logic
grep -c "phases" scripts/train_alphazero.py
# Expected output: 2 (check and iteration)
```

---

## Deployment Checklist

- [x] All files modified correctly
- [x] No syntax errors
- [x] Backward compatible
- [x] Ready for testing
- [x] Documentation complete
- [x] Quick start guide created
- [x] Implementation summary created
- [x] Checklist created

**Status**: ✅ Ready for deployment
