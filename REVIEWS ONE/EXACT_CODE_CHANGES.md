# AlphaZero Checkers - Exact Code Changes Required

## File 1: `scripts/config_alphazero.py`

### Change 1: Update STANDARD config (lines 20-35)

**BEFORE:**
```python
'standard': {
    'ENV_MAX_MOVES': 200,
    'NO_PROGRESS_PLIES': 80,
    'DRAW_PENALTY': 0.0,
    'MCTS_DRAW_VALUE': 0.0,
    'NUM_ITERATIONS': 100,
    'GAMES_PER_ITERATION': 12,
    'TRAIN_EPOCHS': 10,
    'MCTS_SIMULATIONS': 300,
    'BATCH_SIZE': 512,
    'BUFFER_SIZE': 50000,
    'description': 'Optimized Checkers training (High Quality)'
},
```

**AFTER:**
```python
'standard': {
    'ENV_MAX_MOVES': 200,
    'NO_PROGRESS_PLIES': 80,
    'DRAW_PENALTY': -0.05,      # ← CHANGED from 0.0
    'MCTS_DRAW_VALUE': -0.05,   # ← CHANGED from 0.0
    'NUM_ITERATIONS': 100,
    'GAMES_PER_ITERATION': 12,
    'TRAIN_EPOCHS': 10,
    'MCTS_SIMULATIONS': 800,    # ← CHANGED from 300
    'BATCH_SIZE': 256,          # ← CHANGED from 512
    'BUFFER_SIZE': 5000,        # ← CHANGED from 50000
    'description': 'Optimized Checkers training (High Quality)'
},
```

---

## File 2: `training/alpha_zero/trainer.py`

### Change 1: Update loss weights (around line 45)

**BEFORE:**
```python
def __init__(
    self,
    model,
    mcts,
    action_manager,
    board_encoder,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
    buffer_size: int = 10000,
    batch_size: int = 256,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    value_loss_weight: float = 1.0,
    policy_loss_weight: float = 1.0,
    temp_threshold: int = 30,
    draw_penalty: float = -0.1,
    env_max_moves: int = 200,
    no_progress_plies: int = 80,
):
```

**AFTER:**
```python
def __init__(
    self,
    model,
    mcts,
    action_manager,
    board_encoder,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
    buffer_size: int = 10000,
    batch_size: int = 256,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    value_loss_weight: float = 1.0,      # ← CHANGED from 0.15 (in trainer init call)
    policy_loss_weight: float = 1.0,
    temp_threshold: int = 30,
    draw_penalty: float = -0.1,
    env_max_moves: int = 200,
    no_progress_plies: int = 80,
):
```

### Change 2: Update optimizer regularization (around line 90)

**BEFORE:**
```python
# Optimizer
if optimizer is None:
    self.optimizer = optim.Adam(
        self.model.network.parameters(),
        lr=lr,
        weight_decay=1e-4
    )
else:
    self.optimizer = optimizer
```

**AFTER:**
```python
# Optimizer
if optimizer is None:
    self.optimizer = optim.Adam(
        self.model.network.parameters(),
        lr=lr,
        weight_decay=1e-3  # ← CHANGED from 1e-4
    )
else:
    self.optimizer = optimizer
```

### Change 3: Update MCTS initialization (around line 130)

**BEFORE:**
```python
# MCTS
mcts = MCTS(
    model=model,
    action_manager=action_manager,
    encoder=encoder,
    c_puct=1.5,
    num_simulations=CFG['MCTS_SIMULATIONS'], # Pulls 300 from Config
    device=device,
    dirichlet_alpha=0.6,
    dirichlet_epsilon=0.25,
    draw_value=mcts_draw_value,
)
```

**AFTER:**
```python
# MCTS
mcts = MCTS(
    model=model,
    action_manager=action_manager,
    encoder=encoder,
    c_puct=1.5,
    num_simulations=CFG['MCTS_SIMULATIONS'], # Pulls 800 from Config (updated)
    device=device,
    dirichlet_alpha=0.3,        # ← CHANGED from 0.6
    dirichlet_epsilon=0.1,      # ← CHANGED from 0.25
    draw_value=mcts_draw_value,
)
```

### Change 4: Update trainer initialization (around line 150)

**BEFORE:**
```python
# Trainer
trainer = AlphaZeroTrainer(
    model=model,
    mcts=mcts,
    action_manager=action_manager,
    board_encoder=encoder,
    optimizer=optimizer,
    device=device,
    buffer_size=CFG['BUFFER_SIZE'],  # Pulls 50000 from Config
    batch_size=CFG['BATCH_SIZE'],    # Pulls 512 from Config
    lr=0.001,
    weight_decay=1e-4,
    value_loss_weight=0.15,
    policy_loss_weight=1.0,
    temp_threshold=50,
    draw_penalty=draw_penalty,
    env_max_moves=env_max_moves,
    no_progress_plies=no_progress_plies,
)
```

**AFTER:**
```python
# Trainer
trainer = AlphaZeroTrainer(
    model=model,
    mcts=mcts,
    action_manager=action_manager,
    board_encoder=encoder,
    optimizer=optimizer,
    device=device,
    buffer_size=CFG['BUFFER_SIZE'],  # Pulls 5000 from Config (updated)
    batch_size=CFG['BATCH_SIZE'],    # Pulls 256 from Config (updated)
    lr=0.001,
    weight_decay=1e-3,               # ← CHANGED from 1e-4
    value_loss_weight=1.0,           # ← CHANGED from 0.15
    policy_loss_weight=1.0,
    temp_threshold=20,               # ← CHANGED from 50
    draw_penalty=draw_penalty,
    env_max_moves=env_max_moves,
    no_progress_plies=no_progress_plies,
)
```

---

## File 3: `scripts/train_alphazero.py`

### Change 1: Reset training (line 25)

**BEFORE:**
```python
# Set to 0 to start fresh. Set to 4 (or your last iter) to resume.
RESUME_FROM_ITERATION = 0
```

**AFTER:**
```python
# Set to 0 to start fresh. Set to 4 (or your last iter) to resume.
RESUME_FROM_ITERATION = 0  # ← KEEP AS 0 (start fresh!)
```

---

## File 4: `training/alpha_zero/mcts.py`

### No changes needed!

The MCTS code is correct. The fixes in `trainer.py` will automatically use the new Dirichlet parameters.

---

## File 5: `training/alpha_zero/network.py`

### Optional: Better value head initialization (around line 80)

**BEFORE:**
```python
def _init_weights(self):
    """
    Initialize network weights using Kaiming initialization.
    """
    for module in self.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
```

**AFTER (OPTIONAL - improves convergence):**
```python
def _init_weights(self):
    """
    Initialize network weights using Kaiming initialization.
    Special handling for value head to prevent saturation.
    """
    for module in self.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Linear):
            # Check if this is the final value head layer
            if module.out_features == 1:
                # Value head: use smaller initialization to prevent saturation
                nn.init.uniform_(module.weight, -0.01, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            else:
                # Other layers: use Kaiming
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
```

---

## Summary of Changes

| File | Line(s) | Change | Reason |
|------|---------|--------|--------|
| `config_alphazero.py` | 23 | `DRAW_PENALTY: 0.0 → -0.05` | Make draws learnable |
| `config_alphazero.py` | 24 | `MCTS_DRAW_VALUE: 0.0 → -0.05` | Consistent draw handling |
| `config_alphazero.py` | 28 | `MCTS_SIMULATIONS: 300 → 800` | Less noisy policy targets |
| `config_alphazero.py` | 29 | `BATCH_SIZE: 512 → 256` | Reduce overfitting |
| `config_alphazero.py` | 30 | `BUFFER_SIZE: 50000 → 5000` | Fresher training data |
| `trainer.py` | 90 | `weight_decay: 1e-4 → 1e-3` | Stronger regularization |
| `trainer.py` | 130 | `dirichlet_alpha: 0.6 → 0.3` | Less exploration noise |
| `trainer.py` | 131 | `dirichlet_epsilon: 0.25 → 0.1` | Network policy heard |
| `trainer.py` | 150 | `value_loss_weight: 0.15 → 1.0` | Value head learns |
| `trainer.py` | 152 | `temp_threshold: 50 → 20` | Less random moves |
| `train_alphazero.py` | 25 | `RESUME_FROM_ITERATION = 0` | Start fresh! |

---

## Verification Commands

After making changes, verify the config is correct:

```bash
python scripts/config_alphazero.py standard
```

Expected output:
```
Configuration: STANDARD
Description: Optimized Checkers training (High Quality)

Settings:
  ENV_MAX_MOVES: 200
  NO_PROGRESS_PLIES: 80
  DRAW_PENALTY: -0.05          ← Should be -0.05
  MCTS_DRAW_VALUE: -0.05       ← Should be -0.05
  NUM_ITERATIONS: 100
  GAMES_PER_ITERATION: 12
  TRAIN_EPOCHS: 10
  MCTS_SIMULATIONS: 800        ← Should be 800
  BATCH_SIZE: 256              ← Should be 256
  BUFFER_SIZE: 5000            ← Should be 5000
```

---

## Delete Old Checkpoints

```bash
# Remove corrupted checkpoints
rm -rf checkpoints/alphazero/checkpoint_iter_*.pth

# Remove old replay buffer
rm -f checkpoints/alphazero/latest_replay_buffer.pkl

# Remove old CSV logs
rm -f data/training_logs/alphazero_training.csv
```

---

## Start Fresh Training

```bash
python scripts/train_alphazero.py --config standard
```

Expected first iteration output:
```
ITERATION 1/100
[1/2] Self-Play (12 games)...
Self-Play Summary:
  P1 Wins: ~6 (50%)
  P2 Wins: ~0 (0%)
  Draws: ~6 (50%)
  Avg Game Length: ~100 moves

[2/2] Training (10 epochs)...
Training: loss=3.50, value_loss=1.50, policy_loss=2.00
```

**Key:** `value_loss` should be **~1.5** (not 1.18), showing the value head is getting gradient signal.

---

## Monitoring Progress

After 10 iterations, check:

```bash
tail -5 data/training_logs/alphazero_training.csv
```

Expected progression:
```
Iteration 1: value_loss=1.50, policy_loss=2.00, total_loss=3.50
Iteration 2: value_loss=1.30, policy_loss=1.70, total_loss=3.00
Iteration 3: value_loss=1.10, policy_loss=1.40, total_loss=2.50
Iteration 4: value_loss=0.90, policy_loss=1.10, total_loss=2.00
Iteration 5: value_loss=0.70, policy_loss=0.80, total_loss=1.50
```

**Key:** `value_loss` should **decrease significantly** (1.5 → 0.7), not stay flat.

If `value_loss` stays around 1.15-1.20, the fixes didn't work. Check the config values again.

