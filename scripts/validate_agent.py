import sys
import os
import torch
import numpy as np
from tqdm import tqdm  # type: ignore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.game import CheckersEnv
from core.action_manager import ActionManager
from core.board_encoder import CheckersBoardEncoder
from training.alpha_zero.network import AlphaZeroModel
from training.alpha_zero.mcts import MCTS

def run_gauntlet(checkpoint_path, num_games=50):
    print(f"⚔️  THE GAUNTLET: Testing {checkpoint_path} vs RANDOM over {num_games} games...")
    
    # Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    action_manager = ActionManager(device)
    encoder = CheckersBoardEncoder()
    
    model = AlphaZeroModel(action_manager.action_dim, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.network.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Stats
    results = {"win": 0, "loss": 0, "draw": 0}
    
    for i in tqdm(range(num_games)):
        env = CheckersEnv()
        env.reset()
        
        # MCTS Agent plays RED (Player 1)
        # Random plays BLACK (Player -1)
        
        while not env.done:
            if env.current_player == 1:
                # --- AGENT MOVE ---
                mcts = MCTS(model, action_manager, encoder, num_simulations=50, device=device) # Low sims for speed
                probs, _ = mcts.get_action_prob(env, temp=0.1, training=False)
                action_id = int(np.argmax(probs))
                
                # Convert to move
                move_pair = action_manager.get_move_from_id(action_id)
                legal_moves = env.get_legal_moves()
                selected_move = None
                
                # Match move
                for move in legal_moves:
                    start = move[0][0] if isinstance(move, list) else move[0]
                    landing = move[0][1] if isinstance(move, list) else move[1]
                    if (start, landing) == move_pair:
                        selected_move = move
                        break
                
                if selected_move is None:
                    selected_move = legal_moves[0] # Fallback (shouldn't happen)
                    
            else:
                # --- RANDOM MOVE ---
                legal_moves = env.get_legal_moves()
                import random
                selected_move = random.choice(legal_moves)
            
            env.step(selected_move)
        
        # Game Over
        winner = env.winner
        if winner == 1: results["win"] += 1
        elif winner == -1: results["loss"] += 1
        else: results["draw"] += 1
        
    print("\n📊 GAUNTLET RESULTS:")
    print(f"  Wins:  {results['win']} ({results['win']/num_games*100:.1f}%)")
    print(f"  Draws: {results['draw']} ({results['draw']/num_games*100:.1f}%)")
    print(f"  Loss:  {results['loss']} ({results['loss']/num_games*100:.1f}%)")
    print(f"  Non-Loss Rate: {(results['win']+results['draw'])/num_games*100:.1f}%")

if __name__ == "__main__":
    # Update this path to your latest checkpoint
    CHECKPOINT = "checkpoints/alphazero/checkpoint_iter_6.pth" 
    run_gauntlet(CHECKPOINT)