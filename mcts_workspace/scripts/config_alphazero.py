#!/usr/bin/env python3
"""
config_alphazero.py - AlphaZero Configuration Presets

Provides pre-configured training setups for different scenarios:
- Quick test: Fast iteration for debugging
- Standard: Balanced training for development
- Phased: Curriculum learning (Old strategy)
- Madras Local: Resuming high-volume runs on local hardware
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
        'DRAW_PENALTY': 0.0,         # Pure zero-sum
        'MCTS_DRAW_VALUE': 0.0,      # Pure zero-sum
        'NUM_ITERATIONS': 100,
        'GAMES_PER_ITERATION': 12,
        'TRAIN_EPOCHS': 10,
        'MCTS_SIMULATIONS': 800,    
        'BATCH_SIZE': 256,          
        'BUFFER_SIZE': 20000,       # INCREASED: Prevents forgetting wins
        'description': 'Optimized Checkers training (High Quality)'
    },
    
    # YOUR ORIGINAL PHASED STRATEGY (Kept as requested)
    'phased_curriculum': {
        # GLOBAL SETTINGS
        'BUFFER_SIZE': 60000,  
        'BATCH_SIZE': 256,     # Good for RTX 2060
        'TRAIN_EPOCHS': 10,    
        'NUM_ITERATIONS': 200,
        'GAMES_PER_ITERATION': 20, 

        'description': 'Fresh Start Optimized: High Volume -> High Depth',

        'phases': [
            # Phase A
            {
                'name': 'Phase A: Mechanics (Fast & Noisy)',
                'iter_start': 1,
                'iter_end': 25,
                'MCTS_SIMULATIONS': 150, 
                'DIRICHLET_EPSILON': 0.35, 
                'TEMP_THRESHOLD': 30,       
                'NO_PROGRESS_PLIES': 40,    
                'ENV_MAX_MOVES': 80,
                'DRAW_PENALTY': 0.0,
                'MCTS_DRAW_VALUE': 0.0,
                'MCTS_SEARCH_DRAW_BIAS': -0.10, 
            },
            # Phase B
            {
                'name': 'Phase B: Aggression (Anti-Draw)',
                'iter_start': 26,
                'iter_end': 100,
                'MCTS_SIMULATIONS': 800,    
                'DIRICHLET_EPSILON': 0.3,
                'TEMP_THRESHOLD': 30,
                'NO_PROGRESS_PLIES': 60,
                'ENV_MAX_MOVES': 120,
                'DRAW_PENALTY': 0.0,      
                'MCTS_DRAW_VALUE': 0.0,   
                'MCTS_SEARCH_DRAW_BIAS': -0.25, 
            },
            # Phase C
            {
                'name': 'Phase C: Deep Thought (Endgames)',
                'iter_start': 101,
                'iter_end': 200,
                'MCTS_SIMULATIONS': 800,    
                'DIRICHLET_EPSILON': 0.15,
                'TEMP_THRESHOLD': 12,       
                'NO_PROGRESS_PLIES': 100,
                'ENV_MAX_MOVES': 200,       
                'DRAW_PENALTY': 0.0,
                'MCTS_DRAW_VALUE': 0.0,
                'MCTS_SEARCH_DRAW_BIAS': -0.15, 
            },
        ]
    },

    # ====================================================================
    # NEW: LOCAL RESUME CONFIG (For Continuing Cloud Runs on RTX 2060)
    # ====================================================================
    #'madras_local_resume': {
     #   'description': 'Resuming Madras strategy on Local RTX 2060',
        
        # --- VOLUME SETTINGS ---
        #'NUM_ITERATIONS': 500,     # Continue the marathon
        
        # DOWNGRADED FOR LOCAL HARDWARE:
        # 1. 256 Batch (vs 1024/4096 on Cloud) to fit 6GB VRAM
        # 2. 20 Games (vs 64 on Cloud) to fit i7 CPU
        #'BATCH_SIZE': 256,         
       # 'GAMES_PER_ITERATION': 20, 
        
        # KEEP STRATEGY CONSISTENT WITH CLOUD:
        #'MCTS_SIMULATIONS': 600,    # Speed strategy
     #   'C_PUCT': 1.5,             # High exploration
      #  'BUFFER_SIZE': 75000,      # Keep large buffer
       # 'TRAIN_EPOCHS': 10,        # Standard training intensity
        #'LR': 0.002,
        
        # --- LOGIC ---
       # 'DRAW_PENALTY': 0.0,
        #'MCTS_DRAW_VALUE': 0.0,
        #'MCTS_SEARCH_DRAW_BIAS': -0.05,
        
      #  'DIRICHLET_EPSILON': 0.20,
       # 'TEMP_THRESHOLD': 30,
        
       # 'NO_PROGRESS_PLIES': 50,
       # 'ENV_MAX_MOVES': 110,
    #}TILL ITER 142

    'madras_local_resume': {
        'description': 'Phase 2: Wide Vision & Trap Setting (Iter 142+)',
        
        # --- VOLUME ---
        'NUM_ITERATIONS': 500,
        'GAMES_PER_ITERATION': 20,# WAS 20 UNTIL 175 
        
        # --- BRAIN ---
        'MCTS_SIMULATIONS': 800,    # WAS 600 UNTIL 175 
        
        # --- TRAINING ---
        'BATCH_SIZE': 256,         
        'BUFFER_SIZE': 50000,      
        'TRAIN_EPOCHS': 10,        
        'LR': 0.001,               # WAS 0.002 UNTIL 175
        
        # --- MCTS LOGIC (THE HACK) ---
        'C_PUCT': 1.5,             
        'DIRICHLET_ALPHA': 1,    # WAS 0.2 UNTIL 155 AND 1 UNTIL 175 
        'DIRICHLET_EPSILON': 0.25, # Stronger noise to break tunnel vision
        'TEMP_THRESHOLD': 50,# WAS 50 UNTIL 175
        
        # --- AGGRESSION ---
        'DRAW_PENALTY': 0.0,
        'MCTS_DRAW_VALUE': 0.0,
        'MCTS_SEARCH_DRAW_BIAS': -0.30, # Your requested bias
        
        # --- SPEED ---
        'NO_PROGRESS_PLIES': 50,
        'ENV_MAX_MOVES': 110,
    },
    'era9_precision': {
        'description': 'Iter 201+: 1600 Sims, LR 0.0002, PURE ALPHAZERO (No Bias)',
        
        # --- VOLUME (Quality > Quantity, but decent volume) ---
        'NUM_ITERATIONS': 1000,    
        'GAMES_PER_ITERATION': 16, # <--- Bumped to 16. We need data density.
        'BATCH_SIZE': 256,
        'BUFFER_SIZE': 50000,      
        
        # --- BRAIN (MAX POWER) ---
        'MCTS_SIMULATIONS': 1600,  # <--- Grandmaster Depth
        
        # --- PRECISION TRAINING ---
        'TRAIN_EPOCHS': 10,        # <--- Keep at 10. Do NOT increase.
        'LR': 0.0002,              # <--- Surgical Precision.
        
        # --- LOGIC ---
        'C_PUCT': 1.5,
        'DIRICHLET_ALPHA': 0.5,    # <--- Low Noise. Trust the calculation.
        'DIRICHLET_EPSILON': 0.15,
        'TEMP_THRESHOLD': 30,      
        
        # --- PURE ZERO-SUM (The Critic's Fix) ---
        'DRAW_PENALTY': 0.0,
        'MCTS_DRAW_VALUE': 0.0,
        'MCTS_SEARCH_DRAW_BIAS': 0.0, # <--- REMOVED. The Endgame Detector handles this now.
        
        # --- SPEED ---
        'NO_PROGRESS_PLIES': 50,
        'ENV_MAX_MOVES': 110,
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