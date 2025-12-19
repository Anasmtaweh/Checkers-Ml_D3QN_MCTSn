import torch
import time
import argparse
import numpy as np
from checkers_env.env import CheckersEnv
from checkers_agents.random_agent import RandomAgent
from training.common.action_manager import ActionManager
from training.common.board_encoder import CheckersBoardEncoder
from training.d3qn.model import D3QNModel

def evaluate(checkpoint_path, agent_player=1, delay=0.5):
    env = CheckersEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {checkpoint_path}")
    print(f"Agent playing as: Player {agent_player} ({'Red' if agent_player==1 else 'Black'})")

    # Load Components
    action_manager = ActionManager(device=device)
    encoder = CheckersBoardEncoder()
    model = D3QNModel(action_dim=action_manager.action_dim, device=device).to(device)
    
    # Load Weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "online_model_state_dict" in checkpoint:
        model.online.load_state_dict(checkpoint["online_model_state_dict"])
    elif "model_online" in checkpoint:
        model.online.load_state_dict(checkpoint["model_online"])
    else:
        model.online.load_state_dict(checkpoint)
    model.eval()

    opponent = RandomAgent()
    state = env.reset()
    done = False
    info = {}  # Initialize info dict
    
    env.render()
    
    while not done:
        time.sleep(delay)
        current_player = env.current_player
        
        # Is it the Agent's turn?
        if current_player == agent_player:
            print(f"\n🤖 D3QN (Player {agent_player}) Thinking...")
            
            # CRITICAL: If playing as P2, we must flip the board perspective 
            # so the agent sees its own pieces as '1' (Positive)
            # The encoder might handle this, but let's be explicit based on your training logic.
            # Usually training passes 'current_player' to encoder.
            
            state_tensor = encoder.encode(state, player=current_player).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_values = model.online(state_tensor)
            
            # Mask Illegal Moves
            legal_moves = env.get_legal_moves()
            legal_mask = action_manager.make_legal_action_mask(legal_moves).to(device).unsqueeze(0)
            
            masked_q = q_values.clone()
            masked_q[~legal_mask] = -float('inf')
            
            best_action_idx = torch.argmax(masked_q, dim=1).item()
            best_q_val = masked_q[0, best_action_idx].item()
            
            env_action = action_manager.get_move_from_id(int(best_action_idx))
            print(f"Q-Value: {best_q_val:.4f} | Move: {env_action}")
            state, reward, done, info = env.step(env_action)
            
        else:
            print(f"\n🎲 Random Agent (Player {3-agent_player}) Moving...")
            legal_moves = env.get_legal_moves()
            if not legal_moves:
                done = True
                continue
            action, _ = opponent.select_action(state, current_player, legal_moves)
            state, reward, done, info = env.step(action)

        env.render()

    winner = info.get('winner', 0)
    if winner == agent_player:
        print("\n🏆 VICTORY! D3QN Agent Won!")
    elif winner == 0:
        print("\n🤝 DRAW!")
    else:
        print("\n💀 DEFEAT! Random Agent Won.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--player", type=int, default=1, help="Play as 1 (Red) or 2 (Black)")
    parser.add_argument("--speed", type=float, default=0.2, help="Speed")
    args = parser.parse_args()
    
    evaluate(args.path, args.player, args.speed)