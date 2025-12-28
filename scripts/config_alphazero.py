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
        'NUM_ITERATIONS': 10,
        'GAMES_PER_ITERATION': 10,
        'TRAIN_EPOCHS': 5,
        'MCTS_SIMULATIONS': 50,
        'BATCH_SIZE': 128,
        'BUFFER_SIZE': 2000,
        'description': 'Fast testing configuration (1-2 hours)'
    },
    
    # Standard training (overnight)
    'standard': {
        'NUM_ITERATIONS': 100,
        'GAMES_PER_ITERATION': 12,
        'TRAIN_EPOCHS': 10,
        'MCTS_SIMULATIONS': 800,
        'BATCH_SIZE': 256,
        'BUFFER_SIZE': 10000,
        'description': 'Standard training configuration (8-12 hours)'
    },
    
    # Production training (multiple days)
    'production': {
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
