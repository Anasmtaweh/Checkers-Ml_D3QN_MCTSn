
import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from evaluation.fair_tournament import FairTournament
from core.game import CheckersEnv
from training.d3qn.model import DuelingDQN

# Mock Model that always outputs action 0 (Move (7,0)->(6,1) in Canonical P1 view)
# (7,0) is Bottom-Left. (6,1) is Forward-Right.
# For P2 (Flipped), this corresponds to piece at Top-Right (0,7) moving to (1,6).
# Wait.
# Canonical (P1): 7,0 is bottom-left.
# Flipped (P2): 7,0 (Canonical) maps to 0,7 (Absolute).
# So if Model outputs 7,0->6,1.
# This corresponds to Absolute Move 0,7->1,6.
# 0,7 is Top-Right (Black/P2).
# 1,6 is Row 1, Col 6.
# This is a valid P2 move (Top->Down).

class MockModel(torch.nn.Module):
    def __init__(self, action_manager):
        super().__init__()
        self.action_manager = action_manager
        # Find ID for move ((7,0), (6,1))
        self.target_move = ((7,0), (6,1))
        self.target_id = self.action_manager.get_action_id(self.target_move)
        print(f"Mock Model will prefer Canonical Action ID: {self.target_id} -> {self.target_move}")
        
    def forward(self, x, player_side=1):
        # Output high value for target_id
        q = torch.zeros(1, self.action_manager.action_dim)
        if self.target_id >= 0:
            q[0, self.target_id] = 100.0
        return q

def main():
    print("Verifying P2 Inference Fix...")
    
    ft = FairTournament(device="cpu")
    
    # Setup P2 Scenario
    env = CheckersEnv()
    env.reset()
    env.current_player = -1
    
    # We need a board where P2 has a piece at (0,7) and can move to (1,6).
    # Standard board: P2 has pieces at rows 0,1,2.
    # (0,7) has a piece? Yes (Dark square).
    # (1,6) is empty? Yes.
    # So ((0,7), (1,6)) should be a LEGAL Absolute Move.
    
    legal_moves = env.get_legal_moves()
    print(f"Legal Moves (Absolute): {legal_moves}")
    
    target_absolute_move = ((0,7), (1,6))
    if target_absolute_move in legal_moves:
        print(f"Target Move {target_absolute_move} is VALID.")
    else:
        # Maybe my coordinates are wrong?
        # Let's just pick the first legal move and reverse-engineer what the model should output.
        target_absolute_move = legal_moves[0]
        # Flip it to get Canonical Target
        if isinstance(target_absolute_move, list):
             start, end = target_absolute_move[0][0], target_absolute_move[-1][1]
             target_absolute_move = (tuple(start), tuple(end))
        
        print(f"Using Target Absolute Move: {target_absolute_move}")
    
    # Calculate Canonical Target
    # Flip Absolute -> Canonical
    # (r,c) -> (7-r, 7-c)
    can_start = (7-target_absolute_move[0][0], 7-target_absolute_move[0][1])
    can_end = (7-target_absolute_move[1][0], 7-target_absolute_move[1][1])
    canonical_move = (can_start, can_end)
    print(f"Corresponding Canonical Move: {canonical_move}")
    
    # Create Mock Model that prefers this Canonical Move
    mock_model = MockModel(ft.action_manager)
    mock_model.target_move = canonical_move
    mock_model.target_id = ft.action_manager.get_action_id(canonical_move)
    print(f"Mock Model set to output ID {mock_model.target_id}")
    
    # Run get_best_action
    print("\nRunning get_best_action(player=-1)...")
    chosen_move = ft.get_best_action(mock_model, legal_moves, -1)
    
    print(f"\nChosen Move: {chosen_move}")
    
    if isinstance(chosen_move, list):
         chosen_move_tuple = (tuple(chosen_move[0][0]), tuple(chosen_move[-1][1]))
    else:
         chosen_move_tuple = chosen_move
         
    if chosen_move_tuple == target_absolute_move:
        print("✅ SUCCESS! The system correctly mapped Canonical Output -> Absolute Move.")
    else:
        print(f"❌ FAILURE! Expected {target_absolute_move}, got {chosen_move_tuple}")

if __name__ == "__main__":
    main()
