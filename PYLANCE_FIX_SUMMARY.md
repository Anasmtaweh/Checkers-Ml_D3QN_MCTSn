# Pylance Type Error Fix - Ray Remote Function

## Problem

Pylance was reporting a type error on the `@ray.remote()` decorator:

```
No overloads for "__call__" match the provided arguments
Argument of type "(model_state_dict: Unknown, ..., search_draw_bias: Unknown) -> Unknown" 
cannot be assigned to parameter "__function__" of type 
"(T0@__call__, T1@__call__, ..., T9@__call__) -> R@__call__"
Extra parameter "search_draw_bias"
```

## Root Cause

Ray's type stubs only support functions with up to 10 parameters. The `play_game_remote()` function had 11 parameters:
1. model_state_dict
2. action_dim
3. c_puct
4. num_sims
5. temp_threshold
6. dirichlet_alpha
7. dirichlet_epsilon
8. env_max_moves
9. no_progress_plies
10. mcts_draw_value
11. search_draw_bias ← **11th parameter causes error**

## Solution

Refactored the remote function to accept a single dictionary parameter instead of 11 individual parameters:

### Before
```python
@ray.remote(num_gpus=0.22)
def play_game_remote(
    model_state_dict,
    action_dim,
    c_puct,
    num_sims,
    temp_threshold,
    dirichlet_alpha,
    dirichlet_epsilon,
    env_max_moves,
    no_progress_plies,
    mcts_draw_value,
    search_draw_bias,
):
    # ... function body
```

### After
```python
@ray.remote(num_gpus=0.22)
def play_game_remote(params: Dict[str, Any]) -> Dict[str, Any]:
    """Remote worker for parallel self-play games.
    
    Args:
        params: Dictionary containing all parameters
    """
    # Access parameters via params["key"]
```

## Benefits

1. **Fixes Pylance Error**: Single parameter avoids Ray's 10-parameter limit
2. **Better Maintainability**: Easier to add new parameters without changing function signature
3. **Clearer Documentation**: Dictionary keys are self-documenting
4. **Type Safety**: Explicit return type annotation

## Changes Made

### File: training/alpha_zero/trainer.py

1. **Function Signature** (Line ~136):
   - Changed from 11 individual parameters to single `params: Dict[str, Any]`
   - Added explicit return type: `-> Dict[str, Any]`
   - Added comprehensive docstring

2. **Parameter Access** (Throughout function):
   - `model_state_dict` → `params["model_state_dict"]`
   - `action_dim` → `params["action_dim"]`
   - `c_puct` → `params["c_puct"]`
   - `num_sims` → `params["num_sims"]`
   - `temp_threshold` → `params["temp_threshold"]`
   - `dirichlet_alpha` → `params["dirichlet_alpha"]`
   - `dirichlet_epsilon` → `params["dirichlet_epsilon"]`
   - `env_max_moves` → `params["env_max_moves"]`
   - `no_progress_plies` → `params["no_progress_plies"]`
   - `mcts_draw_value` → `params["mcts_draw_value"]`
   - `search_draw_bias` → `params["search_draw_bias"]`

3. **Remote Call** (Line ~245):
   - Changed from passing 11 individual arguments to passing single dictionary
   - Dictionary is constructed with all parameters before calling `.remote()`

## Code Example

### Before
```python
futures.append(
    play_game_remote.remote(
        model_state,
        self.action_manager.action_dim,
        self.mcts.c_puct,
        self.mcts.num_simulations,
        self.temp_threshold,
        self.mcts.dirichlet_alpha,
        self.mcts.dirichlet_epsilon,
        self.env_max_moves,
        self.no_progress_plies,
        getattr(self.mcts, "draw_value", self.draw_penalty),
        getattr(self.mcts, "search_draw_bias", -0.03),
    )
)
```

### After
```python
params = {
    "model_state_dict": model_state,
    "action_dim": self.action_manager.action_dim,
    "c_puct": self.mcts.c_puct,
    "num_sims": self.mcts.num_simulations,
    "temp_threshold": self.temp_threshold,
    "dirichlet_alpha": self.mcts.dirichlet_alpha,
    "dirichlet_epsilon": self.mcts.dirichlet_epsilon,
    "env_max_moves": self.env_max_moves,
    "no_progress_plies": self.no_progress_plies,
    "mcts_draw_value": getattr(self.mcts, "draw_value", self.draw_penalty),
    "search_draw_bias": getattr(self.mcts, "search_draw_bias", -0.03),
}
futures.append(play_game_remote.remote(params))
```

## Verification

The fix:
- ✅ Eliminates Pylance type errors
- ✅ Maintains all functionality
- ✅ Improves code readability
- ✅ Makes it easier to add parameters in the future
- ✅ Follows Ray best practices for complex parameter passing

## Status

✅ **Fixed** - Pylance errors resolved, code is cleaner and more maintainable
