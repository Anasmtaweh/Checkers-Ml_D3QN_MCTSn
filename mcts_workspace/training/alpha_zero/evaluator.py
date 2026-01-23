import os
import time
import torch
import numpy as np
import glob
import random
from shutil import copyfile
import sys

# Ensure we can import from core/training
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.game import CheckersEnv
from core.action_manager import ActionManager
from core.board_encoder import CheckersBoardEncoder
from training.alpha_zero.network import AlphaZeroModel
from training.alpha_zero.mcts import MCTS

MAX_GAME_MOVES = 200

class Arena:
    """
    The Arena: Where models fight to be the 'Best'.
    Now features 'Hall of Fame' validation.
    """
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.action_manager = ActionManager(self.device)
        self.encoder = CheckersBoardEncoder()
        print(f"⚔️  Arena initialized on {self.device}")

    def play_match(self, model_p1, model_p2, num_games=10, mcts_sims=300):
        """
        Play a match between two models.
        Returns: (p1_wins, p2_wins, draws)
        """
        # Note: We use the same simulation count for both players for fairness
        mcts1 = MCTS(model_p1, self.action_manager, self.encoder, 
                     num_simulations=mcts_sims, device=self.device, 
                     dirichlet_epsilon=0.0) # Pure skill, no noise
        
        mcts2 = MCTS(model_p2, self.action_manager, self.encoder, 
                     num_simulations=mcts_sims, device=self.device, 
                     dirichlet_epsilon=0.0)

        results = {'w': 0, 'l': 0, 'd': 0}
        
        for i in range(num_games):
            # Alternate colors every game
            if i % 2 == 0:
                # Game i: P1 is Red (1), P2 is Black (-1)
                winner = self._play_single_game(mcts1, mcts2)
                if winner == 1: results['w'] += 1
                elif winner == -1: results['l'] += 1
                else: results['d'] += 1
            else:
                # Game i+1: P2 is Red (1), P1 is Black (-1)
                winner = self._play_single_game(mcts2, mcts1)
                if winner == 1: results['l'] += 1   # P2 won as Red
                elif winner == -1: results['w'] += 1 # P1 won as Black
                else: results['d'] += 1
            
            # Print dot for progress
            print(".", end="", flush=True)
            
        print() # Newline
        return results['w'], results['l'], results['d']

    def _play_single_game(self, red_mcts, black_mcts):
        env = CheckersEnv(max_moves=MAX_GAME_MOVES)
        state = env.reset()
        done = False
        moves = 0
        info = {'winner': 0}
        
        while not done:
            legal_moves = env.get_legal_moves()
            if env.current_player == 1:
                action_probs, _ = red_mcts.get_action_prob(env, temp=0, training=False)
                action_id = int(np.argmax(action_probs))
                move = red_mcts._get_move_from_action(action_id, legal_moves, player=env.current_player)

            else:
                action_probs, _ = black_mcts.get_action_prob(env, temp=0, training=False)
                action_id = int(np.argmax(action_probs))
                move = black_mcts._get_move_from_action(action_id, legal_moves, player=env.current_player)


            if move is None:
                print(f"⚠️ Illegal/Null move by Player {env.current_player} at move {moves}. Forcing Loss.")
                return -env.current_player
                
            _, _, done, info = env.step(move)
            moves += 1
        
        return info["winner"]

def load_model(path, device, action_dim=None):
    try:
        ckpt = torch.load(path, map_location=device)
        ckpt_action_dim = ckpt.get("action_dim", None)

        if action_dim is None:
            if ckpt_action_dim is not None:
                action_dim = int(ckpt_action_dim)
            else:
                action_dim = ActionManager(device).action_dim

        model = AlphaZeroModel(action_dim, device)
        model.load(path)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def monitor_checkpoints(checkpoint_dir, best_model_path="data/models/best_model.pth", hof_dir="data/hall_of_fame"):
    arena = Arena(device="cuda") 
    
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    os.makedirs(hof_dir, exist_ok=True)
    
    # 1. Initialize System: WAIT for the first model
    if not os.path.exists(best_model_path):
        print(f"👀 Watching {checkpoint_dir} for the first champion...")
        
        while True:
            list_of_files = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
            if list_of_files:
                # Found one! Initialize and break the wait loop
                latest_file = max(list_of_files, key=os.path.getctime)
                print(f"\n👑 Initializing Dynasty with {latest_file}")
                copyfile(latest_file, best_model_path)
                iter_num = latest_file.split('_')[-1].split('.')[0]
                copyfile(latest_file, os.path.join(hof_dir, f"champion_iter_{iter_num}.pth"))
                break
            else:
                # Still empty, wait and retry
                print(".", end="", flush=True)
                time.sleep(10)
    
    current_champion = load_model(best_model_path, arena.device)
    processed_files = set(glob.glob(os.path.join(checkpoint_dir, '*.pth')))
    
    print(f"\n⚔️  Evaluator Active. Waiting for new challengers...")
    
    while True:
        all_files = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
        new_files = [f for f in all_files if f not in processed_files]
        new_files.sort(key=os.path.getmtime)
        
        for challenger_path in new_files:
            chal_name = os.path.basename(challenger_path)
            print(f"\n{'='*60}")
            print(f"🥊 NEW CHALLENGER: {chal_name}")
            print(f"{'='*60}")
            
            challenger = load_model(challenger_path, arena.device)
            if not challenger: continue
            
            # --- STAGE 1: The Title Fight (vs Current Champion) ---
            print(f"  Round 1: vs Current Champion (300 Sims)")
            # UPDATED: mcts_sims=300 to match training intensity
            w1, l1, d1 = arena.play_match(challenger, current_champion, num_games=10, mcts_sims=300)
            
            # --- STAGE 2: The Council of Elders (vs 3 Ghosts) ---
            hof_files = glob.glob(os.path.join(hof_dir, '*.pth'))
            
            w2, l2, d2 = 0, 0, 0
            
            if len(hof_files) > 0:
                council_size = min(3, len(hof_files))
                ghosts = random.sample(hof_files, council_size)
                
                print(f"  Round 2: The Council ({council_size} ghosts)")
                
                # Split 12 games among the council
                games_per_ghost = 12 // council_size
                
                for ghost_path in ghosts:
                    print(f"    - vs {os.path.basename(ghost_path)}")
                    ghost = load_model(ghost_path, arena.device)
                    # UPDATED: mcts_sims=300 here too
                    gw, gl, gd = arena.play_match(challenger, ghost, num_games=games_per_ghost, mcts_sims=300)
                    w2 += gw
                    l2 += gl
                    d2 += gd
            else:
                print("  Round 2: Skipped (Hall of Fame empty)")
            
            # TOTAL SCORES
            total_wins = w1 + w2
            total_losses = l1 + l2
            total_draws = d1 + d2
            total_games = (w1+l1+d1) + (w2+l2+d2)
            
            win_rate = total_wins / total_games if total_games > 0 else 0
            
            print(f"\n📊 AGGREGATE STATS (Total Games: {total_games})")
            print(f"   Wins: {total_wins} | Losses: {total_losses} | Draws: {total_draws}")
            print(f"   Win Rate: {win_rate:.1%}")
            
            if win_rate >= 0.55:
                print(f"🏆 NEW CHAMPION! {chal_name} takes the throne.")
                copyfile(challenger_path, best_model_path)
                iter_num = chal_name.split('_')[-1].split('.')[0]
                copyfile(challenger_path, os.path.join(hof_dir, f"champion_iter_{iter_num}.pth"))
                current_champion = load_model(best_model_path, arena.device)
            else:
                print(f"❌ REJECTED. Win rate {win_rate:.1%} < 55%.")
            
            processed_files.add(challenger_path)
            
        time.sleep(30)

if __name__ == "__main__":
    CHECKPOINT_DIR = "checkpoints/alphazero" 
    BEST_MODEL = "data/models/best_model.pth"
    HOF_DIR = "data/hall_of_fame"
    
    # Create dirs if not exist
    os.makedirs(os.path.dirname(BEST_MODEL), exist_ok=True)
    os.makedirs(HOF_DIR, exist_ok=True)
    
    monitor_checkpoints(CHECKPOINT_DIR, BEST_MODEL, HOF_DIR)