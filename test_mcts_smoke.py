import numpy as np
import torch

from core.game import CheckersEnv
from core.action_manager import ActionManager
from core.board_encoder import CheckersBoardEncoder
from training.alpha_zero.network import AlphaZeroModel
from training.alpha_zero.mcts import MCTS

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = CheckersEnv()
    env.reset()

    action_manager = ActionManager(device)
    encoder = CheckersBoardEncoder()
    model = AlphaZeroModel(action_manager.action_dim, device)

    mcts = MCTS(
        model=model,
        action_manager=action_manager,
        encoder=encoder,
        num_simulations=10,   # keep tiny
        c_puct=1.5,
        device=device,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.0,  # off for determinism
    )

    probs, root = mcts.get_action_prob(env, temp=1.0, training=False)
    assert probs.shape[0] == action_manager.action_dim
    assert np.isfinite(probs).all()
    assert abs(probs.sum() - 1.0) < 1e-5

    legal_moves = env.get_legal_moves()
    aid = int(np.argmax(probs))
    move = mcts._get_move_from_action(aid, legal_moves, player=env.current_player)  # type: ignore
    assert move is not None, "MCTS picked an action that cannot be decoded into a legal move"

    _, _, done, info = env.step(move)
    assert not (done and info.get("winner") == -1 and env.move_count == 0), "Immediate illegal-loss after MCTS move"

    print("OK: MCTS smoke passed")

if __name__ == "__main__":
    main()
