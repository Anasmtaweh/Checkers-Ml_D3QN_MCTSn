import os
import sys
import torch
import numpy as np
import copy
from collections import defaultdict

# FIX PATHS
# Add project root to sys.path so we can import mcts_workspace
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcts_workspace.core.game import CheckersEnv
from mcts_workspace.core.action_manager import ActionManager
from mcts_workspace.core.board_encoder import CheckersBoardEncoder
from mcts_workspace.training.alpha_zero.network import AlphaZeroModel
from mcts_workspace.training.alpha_zero.mcts import MCTS

# --- CONFIG ---
# Resolve path relative to this script to ensure it works regardless of CWD
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
AZ_MODEL_PATH = os.path.join(PROJECT_ROOT, "mcts_workspace/checkpoints/alphazero/checkpoint_iter_237.pth")

SIMULATIONS = 1600
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MATCHES = 10

# ==========================================
# THE ENEMY: CLASSIC MINIMAX BOT
# ==========================================
class MinimaxBot:
    def __init__(self, depth=3):
        self.depth = depth
        self.name = f"Minimax_D{depth}"

    def get_move(self, env):
        _, move = self._minimax(env, self.depth, True, float('-inf'), float('inf'))
        return move

    def _evaluate(self, board):
        # Simple Logic: Kings are worth 3, Men worth 1. Center control bonus.
        score = 0
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece == 0: continue
                
                val = 3 if abs(piece) == 2 else 1
                
                # Center control bias (rows 3,4, cols 2-5)
                if 3 <= r <= 4 and 2 <= c <= 5: val += 0.2
                
                if piece > 0: score += val
                else: score -= val
        return score

    def _minimax(self, env, depth, maximizing, alpha, beta):
        legal_moves = env.get_legal_moves()
        
        # Terminal state or max depth
        if depth == 0 or env.done or not legal_moves:
            return self._evaluate(env.board.board), None

        best_move = legal_moves[0]

        if maximizing:
            max_eval = float('-inf')
            for move in legal_moves:
                clone = copy.deepcopy(env)
                clone.step(move)
                eval_val, _ = self._minimax(clone, depth - 1, False, alpha, beta)
                
                if eval_val > max_eval:
                    max_eval = eval_val
                    best_move = move
                alpha = max(alpha, eval_val)
                if beta <= alpha: break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in legal_moves:
                clone = copy.deepcopy(env)
                clone.step(move)
                eval_val, _ = self._minimax(clone, depth - 1, True, alpha, beta)
                
                if eval_val < min_eval:
                    min_eval = eval_val
                    best_move = move
                beta = min(beta, eval_val)
                if beta <= alpha: break
            return min_eval, best_move

# ==========================================
# THE HERO: ALPHAZERO LOADER
# ==========================================
def load_az(path):
    print(f"Loading model from: {path}")
    manager = ActionManager(DEVICE)
    encoder = CheckersBoardEncoder()
    model = AlphaZeroModel(manager.action_dim, DEVICE)
    checkpoint = torch.load(path, map_location=DEVICE)
    model.network.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, manager, encoder

# ==========================================
# THE ARENA
# ==========================================
def run_benchmark():
    print(f"⚔️  BENCHMARK: AlphaZero (Iter 237) vs Classic Minimax")
    print(f"    AZ Sims: {SIMULATIONS} | Minimax Depth: 3")
    
    az_model, manager, encoder = load_az(AZ_MODEL_PATH)
    minimax = MinimaxBot(depth=3)
    
    wins, losses, draws = 0, 0, 0
    
    for i in range(MATCHES):
        env = CheckersEnv(max_moves=150, no_progress_limit=40)
        env.reset()
        
        # Swap colors every game
        az_is_p1 = (i % 2 == 0)
        p1_name = "AlphaZero" if az_is_p1 else "Minimax"
        p2_name = "Minimax" if az_is_p1 else "AlphaZero"
        
        print(f"\nGame {i+1}: {p1_name} (Red) vs {p2_name} (Black)")
        
        mcts = MCTS(az_model, manager, encoder, num_simulations=SIMULATIONS, c_puct=1.0, device=DEVICE)
        
        while not env.done:
            is_az_turn = (env.current_player == 1 and az_is_p1) or (env.current_player == -1 and not az_is_p1)
            
            if is_az_turn:
                # AZ MOVE (Pure, No Noise)
                with torch.no_grad():
                    probs, _ = mcts.get_action_prob(env, temp=0.0, training=False)
                move = mcts._get_move_from_action(int(np.argmax(probs)), env.get_legal_moves(), env.current_player)
            else:
                # MINIMAX MOVE
                move = minimax.get_move(env)
                
            if not move:
                env.winner = -env.current_player
                env.done = True
                break
                
            env.step(move)
            
        # Result
        if env.winner == 0:
            print("  -> Result: DRAW")
            draws += 1
        elif (env.winner == 1 and az_is_p1) or (env.winner == -1 and not az_is_p1):
            print("  -> Result: ALPHAZERO WINS 🏆")
            wins += 1
        else:
            print("  -> Result: MINIMAX WINS 💀")
            losses += 1

    print("\n" + "="*40)
    print(f"FINAL SCORE vs OLD SCHOOL BOT:")
    print(f"AZ Wins:   {wins}")
    print(f"AZ Losses: {losses}")
    print(f"Draws:     {draws}")
    print("="*40)
    
    if losses > 0:
        print("❌ VERDICT: FAIL. Your model loses to a simple script.")
    elif wins > 5:
        print("✅ VERDICT: PASS. Your model understands fundamental Checkers.")
    else:
        print("⚠️ VERDICT: STALEMATE. Your model is safe but toothless.")

if __name__ == "__main__":
    run_benchmark()