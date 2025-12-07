"""
Utility script to play and print a single DDQN game for quick inspection.
"""

import argparse
from typing import Any, Iterable, Tuple

from checkers_env.env import CheckersEnv
from checkers_agents.ddqn_agent import DDQNAgent
from checkers_agents.random_agent import CheckersRandomAgent
from training.ddqn.evaluation import play_game


def main():
    parser = argparse.ArgumentParser(description="Visualize a single DDQN game")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to DDQN checkpoint for agent A")
    parser.add_argument("--opponent", type=str, default="random", help="Either 'random' or a second checkpoint path")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--max-moves", type=int, default=200, help="Max moves per game")
    args = parser.parse_args()

    env = CheckersEnv()

    ddqn_agent = DDQNAgent(device=args.device)
    ddqn_agent.load_weights(args.checkpoint)

    if args.opponent == "random":
        opponent = CheckersRandomAgent()
    else:
        opponent = DDQNAgent(device=args.device)
        opponent.load_weights(args.opponent)

    print("=== Starting visualization game ===")
    result = play_game(
        env=env,
        agent_first=ddqn_agent,
        agent_second=opponent,
        max_moves=args.max_moves,
        starting_player=1,
        verbose=False,
    )

    # Replay with prints: reset and step while logging
    board = env.reset()
    env.current_player = 1
    player = 1
    info = {}
    print("Initial board:")
    try:
        env.render()
    except Exception:
        print(board)

    move_no = 0
    done = False
    while not done and move_no < args.max_moves:
        moves: Iterable[Any]
        if hasattr(env, "get_legal_moves"):
            moves = list(env.get_legal_moves())
        else:
            moves = list(env.legal_moves())
        if not moves:
            break
        agent = ddqn_agent if player == 1 else opponent
        chosen_move = agent.select_action(board, player, moves) if not isinstance(agent, CheckersRandomAgent) else agent.select_action(env)
        move_no += 1
        print(f"\nMove {move_no} by player {player}: {chosen_move}")

        step_result: Tuple[Any, ...] = env.step(chosen_move)
        if len(step_result) == 5:
            board, next_player, reward, done, info = step_result
        else:
            board, reward, done, info = step_result
            next_player = getattr(env, "current_player", -player)

        try:
            env.render()
        except Exception:
            print(board)
        player = next_player
        if info.get("winner") is not None:
            print(f"Winner reported: {info.get('winner')}")

    print("\n=== Game summary ===")
    print(result)


if __name__ == "__main__":
    main()
