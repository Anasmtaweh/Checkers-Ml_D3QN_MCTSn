import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.alpha_zero.mcts import MCTS, AlphaNode
from core.game import CheckersEnv
from core.action_manager import ActionManager
from core.board_encoder import CheckersBoardEncoder

class MockModel:
    """A Fake Neural Network that targets a specific action."""
    def __init__(self, target_action_id):
        self.target_action_id = target_action_id

    def eval(self): pass
    
    def predict(self, state):
        # Create a fake policy: 100% probability on the LEGAL target action
        policy = torch.zeros(170)
        policy[self.target_action_id] = 1.0
        
        # Fake Value: +0.9 (The player whose turn it is is WINNING)
        # In MCTS, if the Child Node (Opponent) sees +0.9, 
        # the Root Node (Current Player) should see -0.9.
        value = torch.tensor([0.9])
        
        return policy, value

def test_minus_sign_logic():
    print("🧪 TEST: Verifying MCTS Value Flipping & Legal Masking...")
    
    # Setup
    env = CheckersEnv()
    env.reset()
    action_manager = ActionManager(device='cpu')
    encoder = CheckersBoardEncoder()
    
    # 1. Find a GUARANTEED Legal Move to test
    legal_moves = env.get_legal_moves()
    target_move = legal_moves[0]
    target_action_id = action_manager.get_action_id(target_move)
    print(f"  Targeting Legal Action ID: {target_action_id} (Move: {target_move})")

    # 2. Setup Mock Model to favor this specific legal move
    model = MockModel(target_action_id)
    
    mcts = MCTS(model, action_manager, encoder, c_puct=1.0, num_simulations=10)
    
    # Run simulation
    probs, root = mcts.get_action_prob(env, temp=0, training=False)
    
    # --- CHECK 1: Did we pick the Target Move? ---
    best_action = int(np.argmax(probs))
    if best_action == target_action_id:
        print(f"✅ PASS: Agent chose Action {best_action} (Matches Priority).")
    else:
        print(f"❌ FAIL: Agent chose Action {best_action} instead of {target_action_id}.")
        
    # --- CHECK 2: Value Flipping Logic ---
    # Model returns +0.9 for EVERY state.
    # Root (P1) -> expands Child (P2). 
    # Child (P2) evaluates to +0.9 ("I am winning").
    # Backprop should flip this: Root (P1) sees -0.9 ("I am losing").
    
    root_value = root.get_greedy_value()
    # Handle tensor if necessary
    if isinstance(root_value, torch.Tensor):
        root_value = root_value.item()

    print(f"  Root Value: {root_value:.4f}")

    if root_value < 0:
        print("✅ PASS: Root Value is NEGATIVE. Logic is correctly flipping opponent's advantage!")
    else:
        print("❌ FAIL: Root Value is POSITIVE. Logic is NOT flipping (Suicide Bug).")

if __name__ == "__main__":
    test_minus_sign_logic()