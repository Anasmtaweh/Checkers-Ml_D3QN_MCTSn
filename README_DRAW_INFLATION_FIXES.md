# Draw Inflation Fixes - Complete Implementation

## 🎯 Objective

Fix the draw inflation issue where the AlphaZero training system was transitioning from "copying randomness" to "copying long draw-ish search behavior," resulting in:
- Draw rate rising to 83%
- Average game length increasing to 140+ moves
- Policy loss dominating mid-run (~0.7)
- Iteration time increasing to ~1721s

## ✅ Solution: Phased Curriculum Training

A three-phase curriculum that prevents MCTS from finding deep draw loops while gradually increasing search strength as the value head improves.

## 📚 Documentation

### Quick Start
- **[PHASED_CURRICULUM_QUICK_START.md](PHASED_CURRICULUM_QUICK_START.md)** - Start here! Quick reference for running training

### Detailed Explanation
- **[DRAW_INFLATION_FIXES.md](DRAW_INFLATION_FIXES.md)** - Complete analysis of root causes and fixes

### Technical Details
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - How the fixes work technically
- **[EXACT_CODE_CHANGES.md](EXACT_CODE_CHANGES.md)** - Exact code modifications made

### Verification
- **[FIXES_APPLIED_CHECKLIST.md](FIXES_APPLIED_CHECKLIST.md)** - Checklist of all applied fixes

## 🚀 Quick Start

### Start Training with Phased Curriculum
```bash
python scripts/train_alphazero.py --config phased_curriculum
```

### Resume Training
```bash
python scripts/train_alphazero.py --config phased_curriculum --resume 10
```

### View Configuration
```bash
python scripts/config_alphazero.py phased_curriculum
```

## 📊 What Changed

### Phase A: Early Exploration (Iterations 1-10)
- MCTS: 400 sims (down from 800)
- Dirichlet: 0.15 (up from 0.1)
- Temp threshold: 15 (down from 20)
- Max moves: 180 (down from 200)
- No-progress: 60 (down from 80)
- Search draw bias: -0.06 (down from -0.03)

**Expected**: Draw rate ~50%, Avg length ~110 moves

### Phase B: Balanced Growth (Iterations 11-30)
- MCTS: 600 sims
- Dirichlet: 0.10
- Temp threshold: 20
- Max moves: 190
- No-progress: 70
- Search draw bias: -0.05

**Expected**: Draw rate ~35%, Avg length ~100 moves

### Phase C: Full Strength (Iterations 31+)
- MCTS: 800 sims
- Dirichlet: 0.10
- Temp threshold: 20
- Max moves: 200
- No-progress: 80
- Search draw bias: -0.03

**Expected**: Draw rate ~25-30%, Avg length ~100 moves, Win rate >55%

## 🔧 Files Modified

| File | Changes |
|------|---------|
| `scripts/config_alphazero.py` | Added phased_curriculum config with 3 phases |
| `training/alpha_zero/mcts.py` | Added configurable search_draw_bias parameter |
| `training/alpha_zero/trainer.py` | Added policy entropy logging |
| `scripts/train_alphazero.py` | Added phase-based parameter application |

## 📈 Expected Results

### Metrics to Monitor
- **draw_rate**: Should decrease from 83% → ~25-30%
- **avg_game_length**: Should decrease from 140 → ~100 moves
- **value_loss**: Should continue decreasing
- **policy_loss**: Should stabilize around 0.4-0.6
- **policy_entropy**: Should decrease steadily
- **win_rate**: Should move away from 50% (random)

### Timeline
- **Iter 1-5**: Draw rate <60%, games ~110 moves
- **Iter 6-10**: Draw rate ~40-50%, games ~110 moves
- **Iter 11-20**: Draw rate ~30-40%, games ~100 moves
- **Iter 21-30**: Draw rate ~25-35%, games ~100 moves
- **Iter 31+**: Draw rate ~25-30%, games ~100 moves, win rate >55%

## 🎓 How It Works

### Why Phased Curriculum Fixes Draw Inflation

1. **Early weak search (400 sims)**: Doesn't explore deeply enough to find draw basins
2. **High exploration (0.15 Dirichlet)**: Spreads visits beyond first safe option
3. **Early exploitation (temp=15)**: Commits to decisive lines sooner
4. **Short games (180 moves, 60 no-progress)**: Cuts long stalemates
5. **Strong draw bias (-0.06)**: Makes draws unattractive during search
6. **Gradual transition**: Smooth progression as value head improves

### Phase Transitions
- **Phase A → B**: At iteration 11, gradually increase search strength
- **Phase B → C**: At iteration 31, use full MCTS power with well-formed policy/value

## 🔍 Monitoring

### View Training Progress
```bash
# In one terminal
python scripts/train_alphazero.py --config phased_curriculum

# In another terminal
tail -f data/training_logs/alphazero_training.csv
```

### Key Metrics in CSV
- `draw_rate`: Should decrease across phases
- `avg_game_length`: Should stabilize <120 moves
- `p1_win_rate`: Should move away from 50%
- `value_loss`: Should continue decreasing
- `policy_loss`: Should stabilize around 0.4-0.6

## 🛠️ Troubleshooting

### Draw rate still high after Phase A?
```python
# In config_alphazero.py, Phase A section:
'DIRICHLET_EPSILON': 0.20,  # Increase from 0.15
'TEMP_THRESHOLD': 10,        # Decrease from 15
'MCTS_SEARCH_DRAW_BIAS': -0.10,  # Decrease from -0.06
```

### Games too short?
```python
# In config_alphazero.py, Phase A section:
'NO_PROGRESS_PLIES': 70,  # Increase from 60
'ENV_MAX_MOVES': 190,     # Increase from 180
```

### Training too slow?
```python
# In config_alphazero.py, Phase A section:
'MCTS_SIMULATIONS': 300,  # Decrease from 400
```

## 📋 Implementation Checklist

- [x] Configuration system updated with phased_curriculum
- [x] MCTS supports configurable search_draw_bias
- [x] Trainer logs policy entropy
- [x] Training script applies phase parameters per iteration
- [x] Phase transitions are automatic
- [x] Documentation complete
- [x] Quick start guide created
- [x] Technical summary created
- [x] Code changes documented
- [x] Verification checklist created

## 🎯 Success Criteria

Training is successful if:

1. **Draw rate trajectory**
   - Iter 1-5: <60% by iter 5
   - Iter 6-10: ~40-50%
   - Iter 11-20: ~30-40%

2. **Avg game length**
   - <120 by iter 10

3. **Value loss**
   - Continues downward trend (<0.1 by iter 12-15)

4. **Policy loss**
   - Declines and stabilizes (~0.4-0.6 later)

5. **Win rate**
   - Moves away from random oscillation: >55% by iter 15

## 📞 Support

### Documentation Files
1. **PHASED_CURRICULUM_QUICK_START.md** - Quick reference
2. **DRAW_INFLATION_FIXES.md** - Detailed explanation
3. **IMPLEMENTATION_SUMMARY.md** - Technical details
4. **EXACT_CODE_CHANGES.md** - Code modifications
5. **FIXES_APPLIED_CHECKLIST.md** - Verification checklist

### Common Issues
- See PHASED_CURRICULUM_QUICK_START.md for troubleshooting
- See DRAW_INFLATION_FIXES.md for detailed explanation
- See IMPLEMENTATION_SUMMARY.md for technical details

## 🚀 Next Steps

1. **Read** PHASED_CURRICULUM_QUICK_START.md
2. **Run** `python scripts/train_alphazero.py --config phased_curriculum`
3. **Monitor** metrics in data/training_logs/alphazero_training.csv
4. **Adjust** phase parameters if needed
5. **Analyze** results against success criteria

## 📝 References

- **AlphaZero Paper**: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
- **Curriculum Learning**: Self-Paced Learning for Mutual Information Maximization
- **MCTS**: A Survey of Monte Carlo Tree Search Methods and Their Applications

## ✨ Key Features

✅ **Automatic Phase Transitions** - No manual intervention needed
✅ **Configurable Draw Bias** - Allows gradual transition
✅ **Policy Entropy Monitoring** - Detects over-commitment to safe lines
✅ **Backward Compatible** - All new parameters have defaults
✅ **Well Documented** - Complete documentation and guides
✅ **Easy to Use** - Single command to start training

---

**Status**: ✅ All fixes successfully applied and ready for training

**Recommended Config**: `phased_curriculum`

**Quick Start**: `python scripts/train_alphazero.py --config phased_curriculum`

**Documentation**: See files listed above

**Last Updated**: 2024
