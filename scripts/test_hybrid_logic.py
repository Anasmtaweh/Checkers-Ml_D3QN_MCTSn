import os
import sys

import numpy as np
import torch

# --- PATH SETUP ---
# Ensure root is in path so we can import mcts_workspace
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from mcts_workspace.core.action_manager import ActionManager
from mcts_workspace.core.board_encoder import CheckersBoardEncoder
from mcts_workspace.core.game import CheckersEnv
from mcts_workspace.training.alpha_zero.mcts import MCTS, AlphaNode
from mcts_workspace.training.alpha_zero.network import AlphaZeroModel

# --- CONFIG ---
MODEL_PATH = "mcts_workspace/checkpoints/alphazero/checkpoint_iter_239.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIMULATIONS = 800  # Fast but deep enough for testing


# =========================================================
# 🧬 THE HYBRID MCTS (The Proposed Fix)
# =========================================================
class HybridMCTS(MCTS):
    def _evaluate_material(self, board):
        # Heuristic: King = 3, Man = 1
        score = 0
        for row in board:
            for piece in row:
                if piece == 0:
                    continue
                val = 3.0 if abs(piece) == 2 else 1.0
                if piece > 0:
                    score += val
                else:
                    score -= val

        # Normalize to -1 to 1 range (assuming max material diff is ~15)
        # Tanh squashes huge advantages into 1.0 smoothly
        return np.tanh(score / 5.0)

    # We override ONLY the node expansion to inject the material mix
    def _expand_node(self, node: AlphaNode, env: CheckersEnv) -> float:
        board = env.board.get_state()
        player = env.current_player

        # 1. Neural Network Prediction
        state_tensor = self.encoder.encode(
            board, player, force_move_from=env.force_capture_from
        )
        with torch.no_grad():
            policy, value = self.model.predict(state_tensor)

        nn_value = float(value)

        # 2. Material Reality Check
        material_value = self._evaluate_material(board)
        if player == -1:
            material_value = -material_value

        # 3. The Blend: 80% Brain, 20% Math
        blended_value = (0.8 * nn_value) + (0.2 * material_value)

        # Standard MCTS logic below...
        node.is_forced = len(env.get_legal_moves()) == 1
        legal_moves = env.get_legal_moves()
        legal_mask = self.action_manager.make_legal_action_mask(
            legal_moves, player=player
        )
        legal_mask_cpu = legal_mask.detach().cpu()
        p_cpu = policy.detach().cpu() * legal_mask_cpu.float()
        if p_cpu.sum() > 0:
            p_cpu /= p_cpu.sum()
        else:
            p_cpu = legal_mask_cpu.float() / (legal_mask_cpu.sum() + 1e-8)

        for action_id in range(self.action_manager.action_dim):
            if bool(legal_mask_cpu[action_id]):
                child = AlphaNode(
                    prior=float(p_cpu[action_id].item()),
                    parent=node,
                    action_taken=action_id,
                )
                node.children[action_id] = child

        return blended_value


# =========================================================
# ⚔️ THE ARENA
# =========================================================
def run_match():
    print("🔄 Loading Model...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        return

    manager = ActionManager(DEVICE)
    encoder = CheckersBoardEncoder()
    model = AlphaZeroModel(manager.action_dim, DEVICE)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.network.load_state_dict(checkpoint["model_state_dict"])
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    model.eval()

    print(f"🥊 MATCH: Standard AZ (Red) vs Hybrid AZ (Black)")
    print(f"   Simulations: {SIMULATIONS}")

    env = CheckersEnv(max_moves=150, no_progress_limit=40)
    env.reset()

    # Create two brains: Same Model, Different Logic
    mcts_standard = MCTS(
        model=model,
        action_manager=manager,
        encoder=encoder,
        num_simulations=SIMULATIONS,
        c_puct=1.0,
        device=DEVICE,
        dirichlet_epsilon=0.0,
    )
    mcts_hybrid = HybridMCTS(
        model=model,
        action_manager=manager,
        encoder=encoder,
        num_simulations=SIMULATIONS,
        c_puct=1.0,
        device=DEVICE,
        dirichlet_epsilon=0.0,
    )

    while not env.done:
        # P1 (Red) = Standard
        # P2 (Black) = Hybrid
        active_mcts = mcts_standard if env.current_player == 1 else mcts_hybrid
        name = "Standard" if env.current_player == 1 else "Hybrid"

        with torch.no_grad():
            probs, root = active_mcts.get_action_prob(env, temp=0.0, training=False)

        action_id = int(np.argmax(probs))
        move = active_mcts._get_move_from_action(
            action_id, env.get_legal_moves(), env.current_player
        )

        if not move:
            print(f"❌ {name} has no moves! Game Over.")
            env.winner = -env.current_player
            env.done = True
            break

        print(
            f"Move {env.move_count}: {name} moves. (Eval: {root.get_greedy_value():.2f})"
        )
        env.step(move)

    # Result
    if env.winner == 1:
        print("\n🏆 RESULT: STANDARD WINS! (Hybrid Failed)")
    elif env.winner == -1:
        print("\n🏆 RESULT: HYBRID WINS! (Logic Verified)")
    else:
        print("\n🤝 RESULT: DRAW (Both are equal)")


if __name__ == "__main__":
    run_match()
