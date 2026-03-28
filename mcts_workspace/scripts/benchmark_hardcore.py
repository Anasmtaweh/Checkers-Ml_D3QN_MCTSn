import copy
import os
import sys
import time

import numpy as np
import torch

# --- SETUP PATHS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcts_workspace.core.action_manager import ActionManager
from mcts_workspace.core.board_encoder import CheckersBoardEncoder
from mcts_workspace.core.game import CheckersEnv
from mcts_workspace.training.alpha_zero.mcts import MCTS
from mcts_workspace.training.alpha_zero.network import AlphaZeroModel

# --- CONFIG ---
# TEST YOUR BEST MODEL HERE
# Updated to existing checkpoint
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_PATH = os.path.join(
    PROJECT_ROOT, "mcts_workspace/checkpoints/alphazero/checkpoint_iter_239.pth"
)
AZ_SIMS = 1600
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- THE ENEMY: UPGRADED MINIMAX ---
class HardcoreMinimax:
    def __init__(self, depth=4):
        self.depth = depth
        self.name = f"Stockfish_Lite_D{depth}"

    def get_move(self, env):
        # Alpha-Beta Pruning Minimax
        _, move = self._minimax(env, self.depth, True, float("-inf"), float("inf"))
        return move

    def _evaluate(self, board):
        score = 0
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece == 0:
                    continue

                # Material
                base_val = 5.0 if abs(piece) == 2 else 1.0

                # Positional Bonuses
                # 1. Center Control (Middle box)
                if 2 <= r <= 5 and 2 <= c <= 5:
                    base_val += 0.2

                # 2. Back Row Defense (Hard to get Kinged)
                if piece == 1 and r == 0:
                    base_val += 0.5  # P1 home row
                if piece == -1 and r == 7:
                    base_val += 0.5  # P2 home row

                if piece > 0:
                    score += base_val
                else:
                    score -= base_val
        return score

    def _minimax(self, env, depth, maximizing, alpha, beta):
        legal_moves = env.get_legal_moves()

        # Terminal checks
        if depth == 0 or env.done or not legal_moves:
            return self._evaluate(env.board.board), None

        best_move = legal_moves[0]

        if maximizing:
            max_eval = float("-inf")
            # Move ordering? No, keeping it simple/brute for now.
            for move in legal_moves:
                # Simulation optimization: Don't deepcopy if possible, but safe is better
                clone = copy.deepcopy(env)
                clone.step(move)
                eval_val, _ = self._minimax(clone, depth - 1, False, alpha, beta)

                if eval_val > max_eval:
                    max_eval = eval_val
                    best_move = move
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float("inf")
            for move in legal_moves:
                clone = copy.deepcopy(env)
                clone.step(move)
                eval_val, _ = self._minimax(clone, depth - 1, True, alpha, beta)

                if eval_val < min_eval:
                    min_eval = eval_val
                    best_move = move
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
            return min_eval, best_move


# --- LOAD AZ ---
def load_az():
    manager = ActionManager(DEVICE)
    encoder = CheckersBoardEncoder()
    model = AlphaZeroModel(manager.action_dim, DEVICE)
    print(f"🔄 Loading {MODEL_PATH}...")
    model.network.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)["model_state_dict"]
    )
    model.eval()
    return model, manager, encoder


# --- THE ARENA ---
def run_gauntlet():
    model, manager, encoder = load_az()

    # LEVEL 1: DEPTH 4 (Intermediate)
    # LEVEL 2: DEPTH 6 (Hard - Warning: Slow)
    opponents = [
        HardcoreMinimax(depth=10),
        # Uncomment the next line if you have patience (10-20 sec per move)
        # HardcoreMinimax(depth=6)
    ]

    for enemy in opponents:
        print(f"\n{'='*50}")
        print(f"🥊 GAUNTLET: AlphaZero vs {enemy.name}")
        print(f"{ '='*50}")

        # Play 2 Games (Mirror)
        score_az = 0
        score_en = 0

        for i in range(2):
            env = CheckersEnv(max_moves=150, no_progress_limit=40)
            env.reset()

            az_p1 = i == 0
            p1_label = "AlphaZero" if az_p1 else enemy.name
            p2_label = enemy.name if az_p1 else "AlphaZero"

            print(f"Game {i+1}: {p1_label} (Red) vs {p2_label} (Black)")

            # Pure MCTS (No noise)
            # FIXED: Used keyword arguments to match MCTS.__init__ signature correctly
            mcts = MCTS(
                model,
                manager,
                encoder,
                num_simulations=AZ_SIMS,
                c_puct=1.0,
                device=DEVICE,
            )

            while not env.done:
                is_az_turn = (env.current_player == 1 and az_p1) or (
                    env.current_player == -1 and not az_p1
                )

                start = time.time()
                if is_az_turn:
                    # AZ THINKS
                    with torch.no_grad():
                        probs, _ = mcts.get_action_prob(env, temp=0.0, training=False)
                    move = mcts._get_move_from_action(
                        int(np.argmax(probs)), env.get_legal_moves(), env.current_player
                    )
                else:
                    # ENEMY THINKS
                    move = enemy.get_move(env)

                if not move:
                    env.winner = -env.current_player
                    env.done = True
                    break

                env.step(move)

            # Result
            if env.winner == 0:
                print("  -> Result: DRAW 🤝")
                score_az += 0.5
                score_en += 0.5
            elif (env.winner == 1 and az_p1) or (env.winner == -1 and not az_p1):
                print("  -> Result: ALPHAZERO WINS 🏆")
                score_az += 1
            else:
                print(f"  -> Result: {enemy.name} WINS 💀")
                score_en += 1

        print(f"\n🏁 MATCH SCORE: AZ {score_az} - {score_en} {enemy.name}")
        if score_en > 0:
            print("🚨 WARNING: Your model has leaks.")


if __name__ == "__main__":
    run_gauntlet()
