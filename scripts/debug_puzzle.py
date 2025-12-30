import sys
import os
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.game import CheckersEnv
from core.action_manager import ActionManager
from core.board_encoder import CheckersBoardEncoder
from training.alpha_zero.network import AlphaZeroModel
from training.alpha_zero.mcts import MCTS

def test_puzzle(checkpoint_path):
    print("🧩 PUZZLE TEST: Testing Board Vision...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    action_manager = ActionManager(device)
    encoder = CheckersBoardEncoder()
    model = AlphaZeroModel(action_manager.action_dim, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.network.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 1. Setup a custom board (Force a kill)
    # R = Red Piece, B = Black Piece, . = Empty
    # Red at (2,1) can jump Black at (3,2) landing at (4,3)
    env = CheckersEnv()
    env.reset()
    
    # Clear board
    env.board.board[:, :] = 0
    
    # Set up specific scenario
    # RED at (2,1)
    env.board.board[2, 1] = 1
    # BLACK at (3,2) (Target)
    env.board.board[3, 2] = -1
    
    print("  Board State: Red at (2,1), Black at (3,2).")
    print("  Expected: Red MUST jump to (4,3).")
    
    # 2. Ask Agent
    mcts = MCTS(model, action_manager, encoder, num_simulations=100, device=device)
    probs, _ = mcts.get_action_prob(env, temp=0, training=False)
    action_id = int(np.argmax(probs))
    
    # 3. Verify
    move_struct = action_manager.get_move_from_id(action_id)
    print(f"  Agent Chose Move: {move_struct}")
    
    expected_landing = (4, 3)
    
    if move_struct[1] == expected_landing:
        print("✅ PASS: Agent found the capture!")
    else:
        print("❌ FAIL: Agent missed the capture.")

if __name__ == "__main__":
    CHECKPOINT = "checkpoints/alphazero/checkpoint_iter_6.pth"
    test_puzzle(CHECKPOINT)