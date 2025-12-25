import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from core.game import CheckersEnv
from core.action_manager import ActionManager
from core.board_encoder import CheckersBoardEncoder
from training.d3qn.model import D3QNModel, DuelingDQN

def get_best_action(model, board, player, action_manager, encoder, device):
    # Encode board (Canonical)
    state_tensor = encoder.encode(board, player).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get Q values
        # Note: DuelingDQN (from fair_tournament) might be different from D3QNModel
        # But let's assume standard behavior for now or try both
        try:
            q_values = model(state_tensor, player_side=player)
        except:
             q_values = model(state_tensor)
             
    # Get raw best action (before masking)
    best_action_id = int(q_values.argmax().item())
    best_move = action_manager.get_move_from_id(best_action_id)
    
    return best_move, q_values

def main():
    print("Reproducing P2 Inference Issue...")
    
    # Setup
    device = "cpu"
    env = CheckersEnv()
    action_manager = ActionManager(device=device)
    encoder = CheckersBoardEncoder()
    
    # 1. Load a Model (Gen 11 Decisive - P1 WR 46%, P2 WR 30%)
    # Using one from the list if available, or just mocking the behavior if I can't load
    model_path = "agents/d3qn/gen11_decisive.pth"
    if not os.path.exists(model_path):
        # Try to find any .pth file
        import glob
        pths = glob.glob("agents/d3qn/*.pth")
        if pths:
            model_path = pths[0]
        else:
            print("No model found, cannot reproduce with actual weights.")
            return

    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # Extract state dict (copied logic from fair_tournament)
    if isinstance(checkpoint, dict):
        if "model_online" in checkpoint:
            state_dict = checkpoint["model_online"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Determine Architecture
    if "p1_value_fc1.weight" in state_dict:
        action_dim = state_dict["p1_advantage_fc2.weight"].shape[0]
        model = DuelingDQN(action_dim=action_dim, device=device)
    elif "value_fc1.weight" in state_dict:
        action_dim = state_dict["advantage_fc2.weight"].shape[0]
        # Need OldDuelingDQN definition here, but let's try D3QNModel if compatible
        # Or just define OldDuelingDQN class inline
        from training.d3qn.model import DuelingDQN as OldDuelingDQN # Hack if they are same
        # Wait, fair_tournament has a specific OldDuelingDQN class.
        # Let's paste it here to be safe.
        class RealOldDuelingDQN(torch.nn.Module):
            def __init__(self, action_dim, device="cpu"):
                super().__init__()
                self.device = device
                self.conv1 = torch.nn.Conv2d(5, 32, kernel_size=3, padding=1)
                self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
                self.flatten_size = 64 * 8 * 8
                self.fc_norm = torch.nn.LayerNorm(self.flatten_size)
                self.value_fc1 = torch.nn.Linear(self.flatten_size, 512)
                self.value_fc2 = torch.nn.Linear(512, 1)
                self.advantage_fc1 = torch.nn.Linear(self.flatten_size, 512)
                self.advantage_fc2 = torch.nn.Linear(512, action_dim)
                self.to(device)
            def forward(self, x, player_side=1):
                x = torch.nn.functional.relu(self.conv1(x))
                x = torch.nn.functional.relu(self.conv2(x))
                x = torch.nn.functional.relu(self.conv3(x))
                x = x.view(x.size(0), -1)
                x = self.fc_norm(x)
                val = self.value_fc2(torch.nn.functional.relu(self.value_fc1(x)))
                adv = self.advantage_fc2(torch.nn.functional.relu(self.advantage_fc1(x)))
                return val + (adv - adv.mean(dim=1, keepdim=True))
        
        model = RealOldDuelingDQN(action_dim=action_dim, device=device)
    else:
        print("Unknown architecture")
        return

    model.load_state_dict(state_dict)
    model.eval()
    
    # 2. Setup P2 Scenario
    # Initial board: P2 (Black, -1) is at Top (rows 0,1,2). P1 (Red, 1) is at Bottom.
    # P2 to move.
    env.reset()
    # Force current player to P2 (-1)
    env.current_player = -1
    # For P2, legal moves are typically from Row 2 to Row 3 (e.g. (2,1)->(3,0) or (2,1)->(3,2))
    legal_moves = env.get_legal_moves()
    print(f"\nScenario: Game Start, P2's Turn.")
    print(f"Legal Moves (Absolute): {legal_moves}")
    # Expect: [((2, 1), (3, 0)), ((2, 1), (3, 2)), ((2, 3), (3, 2)), ...]
    
    # 3. Ask Agent for Move
    # This uses canonical encoding (flips board so P2 pieces are at bottom)
    chosen_move, q_vals = get_best_action(model, env.board.board, -1, action_manager, encoder, device)
    
    print(f"\nAgent Raw Output (Highest Q-value): {chosen_move}")
    
    # 4. Check Validity
    is_legal = False
    for lm in legal_moves:
        if isinstance(lm, list): # Capture
            if (tuple(lm[0][0]), tuple(lm[-1][1])) == chosen_move:
                is_legal = True
        elif isinstance(lm, tuple):
            if (tuple(lm[0]), tuple(lm[1])) == chosen_move:
                is_legal = True
                
    if is_legal:
        print("✅ The agent output a Valid Absolute Move directly.")
    else:
        print("❌ The agent output an INVALID Absolute Move.")
        
        # Check if it corresponds to a "Canonical" move
        # i.e., if we flip it, is it valid?
        # Flip: (r, c) -> (7-r, 7-c)
        flipped_start = (7 - chosen_move[0][0], 7 - chosen_move[0][1])
        flipped_end = (7 - chosen_move[1][0], 7 - chosen_move[1][1])
        flipped_move = (flipped_start, flipped_end)
        
        print(f"Let's flip the agent's move: {chosen_move} -> {flipped_move}")
        
        is_flipped_legal = False
        for lm in legal_moves:
            if isinstance(lm, list):
                if (tuple(lm[0][0]), tuple(lm[-1][1])) == flipped_move:
                    is_flipped_legal = True
            elif isinstance(lm, tuple):
                if (tuple(lm[0]), tuple(lm[1])) == flipped_move:
                    is_flipped_legal = True
                    
        if is_flipped_legal:
            print("🚀 SUCCESS! The agent output a valid CANONICAL move. We just needed to flip it!")
        else:
            print("❓ The move is invalid even after flipping. The agent might just be confused.")

if __name__ == "__main__":
    main()
