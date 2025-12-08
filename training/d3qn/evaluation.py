"""
Evaluation utilities for D3QN Checkers agent.
"""

from typing import Any, Dict, List, Tuple, Optional

from checkers_env.env import CheckersEnv
from checkers_agents.d3qn_agent import D3QNAgent
from checkers_agents.random_agent import CheckersRandomAgent

Move = Tuple[Tuple[int, int], Tuple[int, int]]


def play_game(
    env: Any,
    agent_first: Any,
    agent_second: Any,
    max_moves: int = 300,
    starting_player: int = 1,
    verbose: bool = False,
) -> Dict[str, Any]:

    board = env.reset()
    env.current_player = starting_player

    first_id = starting_player
    second_id = -starting_player

    moves = 0
    total_reward_first = 0.0
    done = False
    current_player_id = starting_player
    info: Dict[str, Any] = {}

    while not done and moves < max_moves:

        legal_moves = env.get_legal_moves()
        if not legal_moves:
            break

        agent = agent_first if current_player_id == first_id else agent_second

        if isinstance(agent, CheckersRandomAgent):
            move = agent.select_action(env)
        else:
            move = agent.select_action(board, env.current_player, legal_moves)

        if move is None:
            break

        next_state, reward, done, info = env.step(move)

        # The reward from env.step() is from the perspective of the player who just moved.
        # We want total_reward_first to be from the perspective of the first player.
        if current_player_id == first_id:
            total_reward_first += reward
        else:
            total_reward_first -= reward

        board = next_state
        current_player_id = env.current_player
        moves += 1

    return {
        "moves": moves,
        "total_reward": total_reward_first,
        "winner": info.get("winner", None),
        "first_player": first_id,
        "second_player": second_id,
    }


def evaluate_d3qn_vs_random(
    checkpoint_path: str,
    num_episodes: int = 50,
    device: str = "cpu",
    max_moves: int = 300,
    verbose: bool = False,
) -> Dict[str, Any]:

    env = CheckersEnv()

    agent = D3QNAgent(device=device)
    agent.load_weights(checkpoint_path)

    rand = CheckersRandomAgent()

    d3qn_first_wins = 0
    d3qn_second_wins = 0
    draws = 0
    total_moves = 0

    for ep in range(num_episodes):

        starting_player = 1 if ep % 2 == 0 else -1
        af = agent if starting_player == 1 else rand
        as_ = rand if starting_player == 1 else agent

        result = play_game(
            env,
            agent_first=af,
            agent_second=as_,
            max_moves=max_moves,
            starting_player=starting_player,
            verbose=verbose,
        )

        winner = result["winner"]
        total_moves += result["moves"]

        d3qn_is_first = starting_player == 1
        d3qn_player_id = result["first_player"] if d3qn_is_first else result["second_player"]

        if winner == d3qn_player_id:
            if d3qn_is_first:
                d3qn_first_wins += 1
            else:
                d3qn_second_wins += 1
        else:
            draws += 1

    d3qn_total_wins = d3qn_first_wins + d3qn_second_wins
    overall = d3qn_total_wins / num_episodes
    first_rate = d3qn_first_wins / (num_episodes / 2)
    second_rate = d3qn_second_wins / (num_episodes / 2)

    return {
        "overall_win_rate": overall,
        "first_player_win_rate": first_rate,
        "second_player_win_rate": second_rate,
        "draw_rate": draws / num_episodes,
        "d3qn_total_wins": d3qn_total_wins,
        "random_wins": num_episodes - d3qn_total_wins - draws,
        "avg_moves": total_moves / num_episodes,
    }


def evaluate_d3qn_vs_ddqn(
    checkpoint_a: str,
    checkpoint_b: str,
    num_episodes: int = 20,
    device: str = "cpu",
    max_moves: int = 300,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate two D3QN checkpoints against each other, alternating who starts.
    Returns aggregated win/draw statistics and optional per-episode details.
    """
    env = CheckersEnv()
    agent_a = D3QNAgent(device=device)
    agent_a.load_weights(checkpoint_a)
    agent_b = D3QNAgent(device=device)
    agent_b.load_weights(checkpoint_b)

    a_wins = 0
    b_wins = 0
    draws = 0
    total_reward_a = 0.0
    total_moves = 0
    raw_winner_counts: Dict[Any, int] = {}

    for ep in range(num_episodes):
        starting_player = 1 if ep % 2 == 0 else -1
        agent_first, agent_second = (agent_a, agent_b) if starting_player == 1 else (agent_b, agent_a)

        result = play_game(
            env,
            agent_first,
            agent_second,
            max_moves=max_moves,
            starting_player=starting_player,
            verbose=verbose,
        )
        winner = result.get("winner", None)
        raw_winner_counts[winner] = raw_winner_counts.get(winner, 0) + 1

        agent_a_player = result["first_player"] if starting_player == 1 else result["second_player"]
        if winner == agent_a_player:
            a_wins += 1
        elif winner is None or winner == 0:
            draws += 1
        else:
            b_wins += 1

        if starting_player == 1:
            total_reward_a += result["total_reward"]
        else:
            total_reward_a -= result["total_reward"]
        total_moves += result["moves"]

    stats = {
        "num_episodes": num_episodes,
        "agent_a_wins": a_wins,
        "agent_b_wins": b_wins,
        "draws": draws,
        "agent_a_win_rate": a_wins / num_episodes if num_episodes else 0.0,
        "agent_b_win_rate": b_wins / num_episodes if num_episodes else 0.0,
        "draw_rate": draws / num_episodes if num_episodes else 0.0,
        "avg_agent_a_reward": total_reward_a / num_episodes if num_episodes else 0.0,
        "avg_moves": total_moves / num_episodes if num_episodes else 0.0,
        "raw_winner_counts": raw_winner_counts,
    }

    return stats
