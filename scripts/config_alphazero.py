#!/usr/bin/env python3
"""
config_alphazero.py - AlphaZero Configuration Presets

Provides pre-configured training setups for different scenarios:
- Quick test: Fast iteration for debugging
- Standard: Balanced training for development
- Production: Full-scale training for best results
"""

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION PRESETS
# ════════════════════════════════════════════════════════════════════

CONFIGS = {
    # Quick test for debugging (1-2 hours)
    'quick_test': {
        'ENV_MAX_MOVES': 200,
        'NO_PROGRESS_PLIES': 80,
        'DRAW_PENALTY': 0.0,
        'MCTS_DRAW_VALUE': 0.0,
        'NUM_ITERATIONS': 10,
        'GAMES_PER_ITERATION': 10,
        'TRAIN_EPOCHS': 5,
        'MCTS_SIMULATIONS': 50,
        'BATCH_SIZE': 128,
        'BUFFER_SIZE': 2000,
        'description': 'Fast testing configuration (1-2 hours)'
    },
    
    # Standard training (OPTIMIZED FOR RTX 2060 + CHECKERS)
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
    
    # Production training (multiple days)
    'production': {
        'ENV_MAX_MOVES': 200,
        'NO_PROGRESS_PLIES': 80,
        'DRAW_PENALTY': 0.0,
        'MCTS_DRAW_VALUE': 0.0,
        'NUM_ITERATIONS': 500,
        'GAMES_PER_ITERATION': 100,
        'TRAIN_EPOCHS': 20,
        'MCTS_SIMULATIONS': 400,
        'BATCH_SIZE': 512,
        'BUFFER_SIZE': 50000,
        'description': 'Production training configuration (2-3 days)'
    },
    
    # High-quality (for competition)
    'competition': {
        'ENV_MAX_MOVES': 200,
        'NO_PROGRESS_PLIES': 80,
        'DRAW_PENALTY': 0.0,
        'MCTS_DRAW_VALUE': 0.0,
        'NUM_ITERATIONS': 1000,
        'GAMES_PER_ITERATION': 200,
        'TRAIN_EPOCHS': 30,
        'MCTS_SIMULATIONS': 800,
        'BATCH_SIZE': 1024,
        'BUFFER_SIZE': 100000,
        'description': 'High-quality training (1+ week)'
    }
}


def print_config(config_name: str):
    """Print configuration details."""
    if config_name not in CONFIGS:
        print(f"❌ Unknown configuration: {config_name}")
        print(f"Available: {', '.join(CONFIGS.keys())}")
        return
    
    config = CONFIGS[config_name]
    print(f"\n{'='*70}")
    print(f"Configuration: {config_name.upper()}")
    print(f"{'='*70}")
    print(f"Description: {config['description']}")
    print(f"\nSettings:")
    for key, value in config.items():
        if key != 'description':
            print(f"  {key}: {value}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import sys
    
    print("AlphaZero Training Configurations\n")
    
    if len(sys.argv) > 1:
        print_config(sys.argv[1])
    else:
        print("Available configurations:")
        for name, config in CONFIGS.items():
            print(f"\n  {name}:")
            print(f"    {config['description']}")
        
        print("\nUsage: python config_alphazero.py <config_name>")
        print("Example: python config_alphazero.py quick_test")