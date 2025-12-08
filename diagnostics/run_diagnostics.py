import os
import sys

# ---------------------------------------------------------
# Ensure project root is on sys.path so "checkers_env" works
# ---------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from checkers_env.env import CheckersEnv
from checkers_env.rules import CheckersRules
from training.common.board_encoder import CheckersBoardEncoder
from training.d3qn.model import D3QNModel


def banner(msg: str):
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)


# ----------------------------------------------------------------------
# TEST 1: Basic imports and CUDA availability
# ----------------------------------------------------------------------
def test_imports_and_device():
    banner("TEST 1: Imports + Device")
    print("Project root:", ROOT)
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device name:", torch.cuda.get_device_name(0))
    print("✅ Imports & device check OK")


# ----------------------------------------------------------------------
# TEST 2: Env reset and starting position
# ----------------------------------------------------------------------
def test_env_reset():
    banner("TEST 2: Env reset + starting state")
    env = CheckersEnv()
    s0 = env.reset()
    print("State shape:", s0.shape)
    assert isinstance(s0, np.ndarray), "State must be a numpy array"
    assert s0.shape == (8, 8), "Board must be 8x8"
    print("current_player:", env.current_player)
    assert env.current_player in (1, -1)
    moves = env.get_legal_moves()
    print("Number of legal moves at start:", len(moves))
    assert len(moves) > 0, "There should be legal moves at the start position"
    print("✅ Env reset looks OK")


# ----------------------------------------------------------------------
# TEST 3: Legal moves over a few steps (robust to game over)
# ----------------------------------------------------------------------
def test_legal_moves():
    banner("TEST 3: Legal moves over a rollout (with safety)")
    env = CheckersEnv()
    env.reset()

    for step in range(20):
        moves = env.get_legal_moves()
        print(f"Step {step}: got {len(moves)} legal moves")
        assert isinstance(moves, list), "get_legal_moves() must return a list"

        if not moves:
            print("No legal moves (probably game over). Stopping this test here.")
            break

        action = moves[0]
        _, reward, done, info = env.step(action)
        print(f"  took action, reward={reward:.2f}, done={done}, winner={info.get('winner')}")

        if done:
            print("Game finished during test_legal_moves; stopping early.")
            break

    print("✅ Legal-move stepping test completed without crashes")


# ----------------------------------------------------------------------
# TEST 4: Capture sequence sanity (any captures at some random state?)
# ----------------------------------------------------------------------
def test_capture_sequences():
    banner("TEST 4: Capture sequence sanity")
    env = CheckersEnv()
    env.reset()

    found_any_capture = False

    for step in range(50):
        board = env.board.get_state()
        caps = CheckersRules.capture_sequences(board, env.current_player)

        if caps:
            found_any_capture = True
            print(f"Found {len(caps)} capture sequence(s) at step {step}, player {env.current_player}")
            # Print one example
            seq = caps[0]
            print(" Example capture sequence:")
            for s in seq:
                print("   ", s)
            break

        # No capture yet → just play one legal move if possible
        moves = env.get_legal_moves()
        if not moves:
            print("No moves available before finding a capture sequence. Game ended.")
            break
        env.step(moves[0])

    if not found_any_capture:
        print("⚠ No capture sequence found in this random search (can happen, not fatal).")
    else:
        print("✅ Capture generation produced at least one sequence")


# ----------------------------------------------------------------------
# TEST 5: Reward shaping sanity (simple + capture)
# ----------------------------------------------------------------------
def test_reward_shaping():
    banner("TEST 5: Reward shaping sanity")
    env = CheckersEnv()
    env.reset()

    # 1) Simple move: should be around -1 plus small positional terms
    moves = env.get_legal_moves()
    assert moves, "No moves at start (unexpected)"
    s1, r1, d1, info1 = env.step(moves[0])
    print("Simple move reward:", r1, "done:", d1, "winner:", info1.get("winner"))

    # 2) Try to search for a capture and apply it
    for _ in range(40):
        board = env.board.get_state()
        caps = CheckersRules.capture_sequences(board, env.current_player)
        if caps:
            capture = caps[0]
            print("Applying a capture sequence:", capture)
            _, rc, dc, info_c = env.step(capture)
            print("Capture reward:", rc, "done:", dc, "winner:", info_c.get("winner"))
            break
        moves = env.get_legal_moves()
        if not moves:
            print("No moves available while searching for capture (game ended).")
            break
        env.step(moves[0])

    print("✅ Reward shaping calls run without crashes")


# ----------------------------------------------------------------------
# TEST 6: Encoder + D3QNModel forward pass
# ----------------------------------------------------------------------
def test_model_forward_pass():
    banner("TEST 6: Encoder + D3QNModel forward pass")
    env = CheckersEnv()
    state = env.reset()
    encoder = CheckersBoardEncoder()

    encoded = encoder.encode(state, env.current_player)
    print("Encoded board shape:", encoded.shape)  # Expect (C, 8, 8)
    assert encoded.ndim == 3, "Encoded board should be (C, H, W)"

    # Build model and run a forward pass
    action_dim = 4032

    # Ensure tensor
    if isinstance(encoded, np.ndarray):
        dummy_batch = torch.from_numpy(encoded).float().unsqueeze(0)
    else:
        dummy_batch = encoded.float().unsqueeze(0)  # (1, C, H, W)

    model = D3QNModel(action_dim=action_dim, device="cpu")

    with torch.no_grad():
        q_values = model.online(dummy_batch)  # Already (1, C, H, W)

    print("Forward pass OK, Q shape:", q_values.shape)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== CHECKERS-ML DIAGNOSTIC SUITE ===")

    test_imports_and_device()
    test_env_reset()
    test_legal_moves()
    test_capture_sequences()
    test_reward_shaping()
    test_model_forward_pass()

    print("\nAll diagnostics completed.")
