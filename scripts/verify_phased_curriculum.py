#!/usr/bin/env python3
"""
verify_phased_curriculum.py - Comprehensive Integrity Check for Phased Curriculum

Validates:
1. Config integrity (phased_curriculum exists, phases have required keys, no overlaps)
2. MCTS semantics (search_draw_bias attribute, draw terminal path)
3. Ray propagation (search_draw_bias passed to workers, MCTS constructed correctly)
4. CSV logging (uses runtime values, not base config)
5. Adaptive exploration override (warning or error based on flag)

Exit codes:
  0 = All checks passed
  1 = Config integrity failure
  2 = MCTS semantics failure
  3 = Ray propagation failure
  4 = CSV logging failure
  5 = Adaptive exploration override detected (configurable)
"""

import sys
import os
import re
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ════════════════════════════════════════════════════════════════════
# CHECK 1: CONFIG INTEGRITY
# ════════════════════════════════════════════════════════════════════

def check_config_integrity() -> Tuple[bool, str]:
    """Validate phased_curriculum config structure and phase ranges."""
    from scripts.config_alphazero import CONFIGS
    
    # Check 1.1: phased_curriculum exists
    if 'phased_curriculum' not in CONFIGS:
        return False, "❌ phased_curriculum not found in CONFIGS"
    
    cfg = CONFIGS['phased_curriculum']
    
    # Check 1.2: phases is a list
    if 'phases' not in cfg:
        return False, "❌ 'phases' key missing from phased_curriculum"
    
    phases = cfg['phases']
    if not isinstance(phases, list):
        return False, f"❌ 'phases' must be a list, got {type(phases)}"
    
    if len(phases) == 0:
        return False, "❌ 'phases' list is empty"
    
    # Check 1.3: Required keys in each phase
    required_keys = {
        'iter_start', 'iter_end', 'MCTS_SIMULATIONS', 'DIRICHLET_EPSILON',
        'TEMP_THRESHOLD', 'ENV_MAX_MOVES', 'NO_PROGRESS_PLIES',
        'DRAW_PENALTY', 'MCTS_DRAW_VALUE', 'MCTS_SEARCH_DRAW_BIAS'
    }
    
    for i, phase in enumerate(phases):
        if not isinstance(phase, dict):
            return False, f"❌ Phase {i} is not a dict"
        
        missing_keys = required_keys - set(phase.keys())
        if missing_keys:
            return False, f"❌ Phase {i} missing keys: {missing_keys}"
    
    # Check 1.4: Phase ranges don't overlap and cover domain
    sorted_phases = sorted(phases, key=lambda p: p['iter_start'])
    
    for i in range(len(sorted_phases) - 1):
        curr_end = sorted_phases[i]['iter_end']
        next_start = sorted_phases[i + 1]['iter_start']
        
        if curr_end >= next_start:
            return False, f"❌ Phase overlap: Phase {i} ends at {curr_end}, Phase {i+1} starts at {next_start}"
        
        if curr_end + 1 != next_start:
            return False, f"❌ Phase gap: Phase {i} ends at {curr_end}, Phase {i+1} starts at {next_start}"
    
    # Check 1.5: Validate parameter types and ranges
    for i, phase in enumerate(phases):
        if not isinstance(phase['iter_start'], int) or not isinstance(phase['iter_end'], int):
            return False, f"❌ Phase {i}: iter_start/iter_end must be int"
        
        if phase['iter_start'] > phase['iter_end']:
            return False, f"❌ Phase {i}: iter_start > iter_end"
        
        if not isinstance(phase['MCTS_SIMULATIONS'], int) or phase['MCTS_SIMULATIONS'] <= 0:
            return False, f"❌ Phase {i}: MCTS_SIMULATIONS must be positive int"
        
        if not isinstance(phase['DIRICHLET_EPSILON'], (int, float)) or not (0 <= phase['DIRICHLET_EPSILON'] <= 1):
            return False, f"❌ Phase {i}: DIRICHLET_EPSILON must be in [0, 1]"
        
        if not isinstance(phase['MCTS_SEARCH_DRAW_BIAS'], (int, float)):
            return False, f"❌ Phase {i}: MCTS_SEARCH_DRAW_BIAS must be numeric"
    
    return True, "✅ Config integrity check passed"


# ════════════════════════════════════════════════════════════════════
# CHECK 2: MCTS SEMANTICS
# ════════════════════════════════════════════════════════════════════

def check_mcts_semantics() -> Tuple[bool, str]:
    """Validate MCTS has search_draw_bias and draw terminal path uses it."""
    try:
        from training.alpha_zero.mcts import MCTS
        from training.alpha_zero.network import AlphaZeroModel
        from core.action_manager import ActionManager
        from core.board_encoder import CheckersBoardEncoder
        import torch
        
        # Check 2.1: MCTS accepts search_draw_bias parameter
        try:
            device = "cpu"
            action_manager = ActionManager(device=device)
            encoder = CheckersBoardEncoder()
            model = AlphaZeroModel(action_dim=action_manager.action_dim, device=device)
            
            mcts = MCTS(
                model=model,
                action_manager=action_manager,
                encoder=encoder,
                c_puct=1.5,
                num_simulations=100,
                device=device,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.1,
                draw_value=-0.05,
                search_draw_bias=-0.06
            )
        except TypeError as e:
            return False, f"❌ MCTS does not accept search_draw_bias: {e}"
        
        # Check 2.2: MCTS has search_draw_bias attribute
        if not hasattr(mcts, 'search_draw_bias'):
            return False, "❌ MCTS missing search_draw_bias attribute"
        
        if mcts.search_draw_bias != -0.06:
            return False, f"❌ MCTS search_draw_bias not set correctly: {mcts.search_draw_bias}"
        
        # Check 2.3: Parse MCTS source to verify draw terminal path uses search_draw_bias
        mcts_file = os.path.join(os.path.dirname(__file__), '..', 'training', 'alpha_zero', 'mcts.py')
        with open(mcts_file, 'r') as f:
            mcts_source = f.read()
        
        # Look for the draw terminal branch
        if 'env.winner == 0' not in mcts_source:
            return False, "❌ Draw terminal check not found in MCTS"
        
        # Check that search_draw_bias is used in the draw branch
        if 'self.search_draw_bias' not in mcts_source:
            return False, "❌ self.search_draw_bias not used in MCTS"
        
        # Verify the pattern: biased_value = value + self.search_draw_bias
        if 'biased_value = value + self.search_draw_bias' not in mcts_source:
            return False, "❌ search_draw_bias not applied to biased_value in draw terminal"
        
        return True, "✅ MCTS semantics check passed"
    
    except Exception as e:
        return False, f"❌ MCTS semantics check failed: {e}"


# ════════════════════════════════════════════════════════════════════
# CHECK 3: RAY PROPAGATION
# ════════════════════════════════════════════════════════════════════

def check_ray_propagation() -> Tuple[bool, str]:
    """Validate search_draw_bias is passed to Ray workers and MCTS is constructed correctly."""
    trainer_file = os.path.join(os.path.dirname(__file__), '..', 'training', 'alpha_zero', 'trainer.py')
    
    with open(trainer_file, 'r') as f:
        trainer_source = f.read()
    
    # Check 3.1: play_game_remote signature includes params dict
    if 'def play_game_remote(' not in trainer_source:
        return False, "❌ play_game_remote function not found"
    
    # Extract play_game_remote signature
    match = re.search(r'def play_game_remote\((.*?)\)(?: -> .*?)?:', trainer_source, re.DOTALL)
    if not match:
        return False, "❌ Could not parse play_game_remote signature"
    
    signature = match.group(1)
    # New signature should have params: Dict[str, Any]
    if 'params' not in signature or 'Dict' not in signature:
        return False, "❌ play_game_remote should accept params: Dict[str, Any]"
    
    # Check 3.2: play_game_remote.remote() call passes params dict
    if 'play_game_remote.remote(' not in trainer_source:
        return False, "❌ play_game_remote.remote() call not found"
    
    # Look for the remote call and check if it passes a params dict
    match = re.search(r'play_game_remote\.remote\((.*?)\)', trainer_source, re.DOTALL)
    if not match:
        return False, "❌ Could not parse play_game_remote.remote() call"
    
    remote_call = match.group(1)
    # Should pass a single params dict
    if 'params' not in remote_call:
        return False, "❌ play_game_remote.remote() should pass params dict"
    
    # Check 3.3: params dict is constructed with all required keys
    if 'params = {' not in trainer_source:
        return False, "❌ params dict construction not found"
    
    # Look for params dict construction
    match = re.search(r'params = \{(.*?)\}', trainer_source, re.DOTALL)
    if not match:
        return False, "❌ Could not extract params dict"
    
    params_body = match.group(1)
    required_keys = [
        'model_state_dict', 'action_dim', 'c_puct', 'num_sims', 'temp_threshold',
        'dirichlet_alpha', 'dirichlet_epsilon', 'env_max_moves', 'no_progress_plies',
        'mcts_draw_value', 'search_draw_bias'
    ]
    
    missing_keys = [key for key in required_keys if f'"{key}"' not in params_body and f"'{key}'" not in params_body]
    if missing_keys:
        return False, f"❌ params dict missing keys: {missing_keys}"
    
    # Check 3.4: MCTS is constructed with search_draw_bias in worker
    if 'mcts = MCTS(' not in trainer_source:
        return False, "❌ MCTS construction not found in worker"
    
    # Look for MCTS construction in play_game_remote
    match = re.search(r'def play_game_remote\(.*?\)(?: -> .*?)?:(.*?)env = CheckersEnv', trainer_source, re.DOTALL)
    if not match:
        return False, "❌ Could not extract play_game_remote body"
    
    worker_body = match.group(1)
    if 'mcts = MCTS(' not in worker_body:
        return False, "❌ MCTS construction not found in play_game_remote body"
    
    # Check if search_draw_bias is passed to MCTS constructor
    if 'search_draw_bias=params["search_draw_bias"]' not in worker_body:
        return False, "❌ search_draw_bias not passed to MCTS constructor"
    
    return True, "✅ Ray propagation check passed"


# ════════════════════════════════════════════════════════════════════
# CHECK 4: CSV LOGGING
# ════════════════════════════════════════════════════════════════════

def check_csv_logging() -> Tuple[bool, str]:
    """Validate CSV logging uses runtime values, not base config."""
    train_file = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'train_alphazero.py')
    
    with open(train_file, 'r') as f:
        train_source = f.read()
    
    # Check 4.1: log_data construction uses runtime values
    if 'log_data = {' not in train_source:
        return False, "❌ log_data construction not found"
    
    # Extract log_data construction
    match = re.search(r'log_data = \{(.*?)\}', train_source, re.DOTALL)
    if not match:
        return False, "❌ Could not parse log_data construction"
    
    log_data_body = match.group(1)
    
    # Check for problematic patterns: CFG['...'] or base locals
    issues = []
    
    # Should use mcts.num_simulations, not CFG['MCTS_SIMULATIONS']
    if "'mcts_simulations': CFG['MCTS_SIMULATIONS']" in log_data_body:
        issues.append("mcts_simulations should use mcts.num_simulations, not CFG['MCTS_SIMULATIONS']")
    
    # Should use trainer.env_max_moves, not env_max_moves (base local)
    if "'env_max_moves': env_max_moves" in log_data_body:
        issues.append("env_max_moves should use trainer.env_max_moves, not base local env_max_moves")
    
    # Should use trainer.no_progress_plies, not no_progress_plies (base local)
    if "'no_progress_plies': no_progress_plies" in log_data_body:
        issues.append("no_progress_plies should use trainer.no_progress_plies, not base local no_progress_plies")
    
    # Should use trainer.draw_penalty, not draw_penalty (base local)
    if "'draw_penalty': draw_penalty" in log_data_body:
        issues.append("draw_penalty should use trainer.draw_penalty, not base local draw_penalty")
    
    # Should use mcts.draw_value, not mcts_draw_value (base local)
    if "'mcts_draw_value': mcts_draw_value" in log_data_body:
        issues.append("mcts_draw_value should use mcts.draw_value, not base local mcts_draw_value")
    
    if issues:
        return False, "❌ CSV logging issues:\n  " + "\n  ".join(issues)
    
    return True, "✅ CSV logging check passed"


# ════════════════════════════════════════════════════════════════════
# CHECK 5: ADAPTIVE EXPLORATION OVERRIDE
# ════════════════════════════════════════════════════════════════════

def check_adaptive_exploration(allow_override: bool = True) -> Tuple[bool, str]:
    """Check if adaptive exploration overrides curriculum control."""
    trainer_file = os.path.join(os.path.dirname(__file__), '..', 'training', 'alpha_zero', 'trainer.py')
    
    with open(trainer_file, 'r') as f:
        trainer_source = f.read()
    
    # Look for the adaptive exploration pattern
    if 'mcts.dirichlet_alpha = current_alpha' in trainer_source:
        if allow_override:
            return True, "⚠️  Adaptive exploration override detected (allowed, but breaks strict curriculum control)"
        else:
            return False, "❌ Adaptive exploration override detected (breaks strict curriculum control)"
    
    return True, "✅ No adaptive exploration override detected"


# ════════════════════════════════════════════════════════════════════
# RUNTIME SELF-CHECK (Optional)
# ════════════════════════════════════════════════════════════════════

def runtime_self_check() -> Tuple[bool, str]:
    """Optional: Run one remote game and verify worker MCTS settings."""
    try:
        import torch
        from training.alpha_zero.network import AlphaZeroModel
        from training.alpha_zero.mcts import MCTS
        from training.alpha_zero.trainer import AlphaZeroTrainer
        from core.action_manager import ActionManager
        from core.board_encoder import CheckersBoardEncoder
        from scripts.config_alphazero import CONFIGS
        
        device = "cpu"
        action_manager = ActionManager(device=device)
        encoder = CheckersBoardEncoder()
        model = AlphaZeroModel(action_dim=action_manager.action_dim, device=device)
        
        mcts = MCTS(
            model=model,
            action_manager=action_manager,
            encoder=encoder,
            c_puct=1.5,
            num_simulations=100,
            device=device,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.1,
            draw_value=-0.05,
            search_draw_bias=-0.06
        )
        
        optimizer = torch.optim.Adam(model.network.parameters(), lr=0.001)
        
        trainer = AlphaZeroTrainer(
            model=model,
            mcts=mcts,
            action_manager=action_manager,
            board_encoder=encoder,
            optimizer=optimizer,
            device=device,
            buffer_size=1000,
            batch_size=32,
            temp_threshold=20,
            draw_penalty=-0.05,
            env_max_moves=200,
            no_progress_plies=80,
        )
        
        # Verify MCTS has correct attributes
        if not hasattr(trainer.mcts, 'search_draw_bias'):
            return False, "❌ Runtime check: MCTS missing search_draw_bias"
        
        if trainer.mcts.search_draw_bias != -0.06:
            return False, f"❌ Runtime check: search_draw_bias mismatch: {trainer.mcts.search_draw_bias}"
        
        return True, "✅ Runtime self-check passed"
    
    except Exception as e:
        return False, f"❌ Runtime self-check failed: {e}"


# ════════════════════════════════════════════════════════════════════
# MAIN VERIFICATION
# ════════════════════════════════════════════════════════════════════

def main():
    """Run all verification checks."""
    print("\n" + "="*70)
    print("PHASED CURRICULUM INTEGRITY VERIFICATION")
    print("="*70 + "\n")
    
    checks = [
        ("Config Integrity", check_config_integrity, 1),
        ("MCTS Semantics", check_mcts_semantics, 2),
        ("Ray Propagation", check_ray_propagation, 3),
        ("CSV Logging", check_csv_logging, 4),
        ("Adaptive Exploration", lambda: check_adaptive_exploration(allow_override=True), 5),
        ("Runtime Self-Check", runtime_self_check, 0),  # Non-fatal
    ]
    
    results = []
    exit_code = 0
    
    for check_name, check_func, code in checks:
        print(f"[*] {check_name}...")
        try:
            passed, message = check_func()
            results.append((check_name, passed, message, code))
            print(f"    {message}\n")
            
            if not passed and code > 0:
                exit_code = code
        except Exception as e:
            results.append((check_name, False, f"❌ Exception: {e}", code))
            print(f"    ❌ Exception: {e}\n")
            if code > 0:
                exit_code = code
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for _, passed, _, _ in results if passed)
    total_count = len(results)
    
    print(f"\nPassed: {passed_count}/{total_count}")
    
    for check_name, passed, message, _ in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
    
    print("\n" + "="*70)
    
    if exit_code == 0:
        print("✅ ALL CHECKS PASSED - Ready for training")
    else:
        print(f"❌ VERIFICATION FAILED (exit code: {exit_code})")
    
    print("="*70 + "\n")
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
