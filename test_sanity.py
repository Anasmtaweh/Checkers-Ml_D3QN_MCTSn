import numpy as np

from core.rules import CheckersRules
from core.game import CheckersEnv

def empty_board():
    return np.zeros((8, 8), dtype=np.int8)

def test_capture_steps_two_directions():
    """
    A single man can have TWO different capture landings.
    This test catches the 'early return inside loop' bug.
    """
    b = empty_board()
    # Player 1 man at (2, 3)
    b[2, 3] = 1
    # Two opponents adjacent diagonals
    b[3, 2] = -1
    b[3, 4] = -1
    # Landing squares empty: (4,1) and (4,5)
    steps = CheckersRules.capture_steps(b, player=1)
    assert len(steps) == 2, f"Expected 2 capture steps, got {len(steps)}: {steps}"

def test_env_forced_chain_behavior():
    """
    Ensure:
    - capture step returns continue=True and does NOT switch player
    - legal moves are restricted to forced_from
    """
    env = CheckersEnv()
    env.reset()

    # Overwrite board for a forced 2-jump chain for player 1:
    # 1 at (2,1) jumps -1 at (3,2) to (4,3), then jumps -1 at (5,4) to (6,5)
    env.board.board[:] = 0
    env.current_player = 1
    env.force_capture_from = None
    env.done = False
    env.winner = 0

    env.board.board[2, 1] = 1
    env.board.board[3, 2] = -1
    env.board.board[5, 4] = -1

    # First capture step
    legal1 = env.get_legal_moves()
    assert len(legal1) >= 1 and len(legal1[0]) == 3, f"Expected capture steps, got: {legal1}"

    step1 = ((2, 1), (4, 3), (3, 2))
    s, r, done, info = env.step([step1])

    assert done is False
    assert info["continue"] is True, f"Expected continue=True, got {info}"
    assert env.current_player == 1, "Player must not switch during forced capture chain"
    assert env.force_capture_from == (4, 3), f"Expected forced_from=(4,3), got {env.force_capture_from}"

    # Now only captures from (4,3) should be legal
    legal2 = env.get_legal_moves()
    starts = set(m[0] for m in legal2)  # each move is (start, landing, jumped)
    assert starts == {(4, 3)}, f"Expected only forced piece to move, got starts={starts}, legal={legal2}"

    # Second capture step should end chain and then switch player
    step2 = ((4, 3), (6, 5), (5, 4))
    s, r, done, info = env.step([step2])
    assert env.current_player == -1, "After finishing chain, player must switch"
    assert env.force_capture_from is None, "forced_from must clear when chain ends"

if __name__ == "__main__":
    test_capture_steps_two_directions()
    test_env_forced_chain_behavior()
    print("OK: sanity tests passed")
