# Quick Reference: Fixes Status & Buffer Strategy

## ✅ ALL FIXES APPLIED - READY TO TRAIN

### Configuration Fixes (scripts/config_alphazero.py)
```
✅ DRAW_PENALTY: 0.0 → -0.05
✅ MCTS_DRAW_VALUE: 0.0 → -0.05
✅ MCTS_SIMULATIONS: 300 → 800
✅ BATCH_SIZE: 512 → 256
✅ BUFFER_SIZE: 50000 → 5000
```

### Trainer Fixes (training/alpha_zero/trainer.py)
```
✅ weight_decay: 1e-4 → 1e-3
✅ value_loss_weight: 0.15 → 1.0
✅ dirichlet_alpha: 0.6 → 0.3
✅ dirichlet_epsilon: 0.25 → 0.1
✅ temp_threshold: 50 → 20
```

### Training Script (scripts/train_alphazero.py)
```
✅ RESUME_FROM_ITERATION: 0 (start fresh)
```

---

## 📈 Buffer Size Strategy

### When to Increase

| Iteration | Buffer Size | Trigger | Metric Target |
|-----------|-------------|---------|---------------|
| 1-30 | 5000 | Current | value_loss < 0.3 |
| 31-60 | 10000 | After iter 30 | value_loss < 0.1 |
| 61-100 | 20000 | After iter 60 | win_rate > 70% |
| 101+ | 50000 | After iter 100 | win_rate > 85% |

### How to Increase

**Edit `scripts/config_alphazero.py`:**
```python
'standard': {
    'BUFFER_SIZE': 10000,  # Change this
    'BATCH_SIZE': 256,     # Keep same
    # ... rest of config ...
}
```

**Then restart training:**
```bash
python scripts/train_alphazero.py --config standard --resume 30
```

---

## 🎯 Success Metrics

### Iteration 5
- ✅ value_loss: 1.5 → <1.0
- ✅ policy_loss: 2.0 → <1.0
- ✅ total_loss: 3.5 → <2.0

### Iteration 10
- ✅ value_loss: <0.5
- ✅ win_rate: >52%
- ✅ draw_rate: <45%

### Iteration 20
- ✅ value_loss: ~0.2
- ✅ win_rate: >60%
- ✅ draw_rate: <30%

### Iteration 30
- ✅ value_loss: <0.1
- ✅ win_rate: >70%
- ✅ draw_rate: <20%
- **→ INCREASE BUFFER TO 10000**

---

## 🚀 Start Training

```bash
python scripts/train_alphazero.py --config standard
```

**Monitor progress:**
```bash
tail -1 data/training_logs/alphazero_training.csv
```

---

## ⚠️ If Something Goes Wrong

### Value loss still flat?
- Check: `value_loss_weight=1.0` in trainer.py
- Check: `DRAW_PENALTY=-0.05` in config

### Win rate still random?
- Check: `dirichlet_alpha=0.3` in trainer.py
- Check: `BUFFER_SIZE=5000` in config

### Training crashes?
- Check: `BATCH_SIZE=256, BUFFER_SIZE=5000` ratio
- Check: GPU memory (should be <6GB)

---

## 📊 Expected Timeline

```
Week 1 (Iter 1-30):   Network learns basics
Week 2 (Iter 31-60):  Network improves (increase buffer)
Week 3 (Iter 61-100): Agent becomes strong (increase buffer)
Week 4+ (Iter 101+):  Elite agent (increase buffer)
```

---

**You're all set! Start training now. 🚀**

