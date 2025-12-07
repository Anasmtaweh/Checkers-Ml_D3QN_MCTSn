__all__ = [
    "play_game",
    "evaluate_ddqn_vs_random",
    "evaluate_ddqn_vs_ddqn",
]

from typing import Any, Dict, List, Optional, Tuple

from checkers_env.env import CheckersEnv
from checkers_agents.ddqn_agent import DDQNAgent
from checkers_agents.random_agent import CheckersRandomAgent

Move = Tuple[Tuple[int, int], Tuple[int, int]]


def _normalize_move(mv: Any) -> Optional[Move]:
    if not isinstance(mv, (list, tuple)):
        return None
    if len(mv) in (2, 3) and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in mv[:2]):
        return (tuple(mv[0]), tuple(mv[1]))
    return None


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

    current_player = starting_player
    first_player = starting_player
    second_player = -first_player

    moves = 0
    total_reward_first = 0.0
    done = False
    info: Dict[str, Any] = {}

    while not done and moves < max_moves:
        legal_moves = env.get_legal_moves() if hasattr(env, "get_legal_moves") else env.legal_moves()
        if not legal_moves:
            break

        agent = agent_first if current_player == first_player else agent_second

        if isinstance(agent, CheckersRandomAgent):
            chosen_move = agent.select_action(env)
        else:
            chosen_move = agent.select_action(board, current_player, legal_moves, greedy=True)

        if chosen_move is None:
            break

        step = env.step(chosen_move)
        if len(step) == 5:
            next_board, next_player, reward, done, info = step
        else:
            next_board, reward, done, info = step
            next_player = env.current_player

        reward_first_persp = reward if current_player == first_player else -reward
        total_reward_first += reward_first_persp

        board = next_board
        current_player = next_player
        moves += 1

    return {
        "moves": moves,
        "total_reward": total_reward_first,
        "winner": info.get("winner", None),
        "max_moves_reached": moves >= max_moves,
        "first_player": first_player,
        "second_player": second_player,
    }


def evaluate_ddqn_vs_random(
    checkpoint_path: str,
    num_episodes: int = 50,
    device: str = "cpu",
    max_moves: int = 300,
    verbose: bool = False,
    return_episode_stats: bool = False,
) -> Dict[str, Any]:

    env = CheckersEnv()
    ddqn_agent = DDQNAgent(device=device)
    ddqn_agent.load_weights(checkpoint_path)
    random_agent = CheckersRandomAgent()

    ddqn_wins = random_wins = draws = 0
    total_reward = 0.0
    total_moves = 0

    episode_details: List[Dict[str, Any]] = []

    for ep in range(num_episodes):
        starting_player = 1 if ep % 2 == 0 else -1
        agent_first = ddqn_agent if starting_player == 1 else random_agent
        agent_second = random_agent if starting_player == 1 else ddqn_agent

        result = play_game(env, agent_first, agent_second, max_moves, starting_player, verbose)

        winner = result["winner"]
        if winner == result["first_player"]:
            ddqn_wins += 1
        elif winner == result["second_player"]:
            random_wins += 1
        else:
            draws += 1

        total_reward += result["total_reward"] if starting_player == 1 else -result["total_reward"]
        total_moves += result["moves"]

        if return_episode_stats:
            episode_details.append(result)

    stats = {
        "num_episodes": num_episodes,
        "ddqn_wins": ddqn_wins,
        "random_wins": random_wins,
        "draws": draws,
        "ddqn_win_rate": ddqn_wins / num_episodes,
        "random_win_rate": random_wins / num_episodes,
        "draw_rate": draws / num_episodes,
        "avg_ddqn_reward": total_reward / num_episodes,
        "avg_moves": total_moves / num_episodes,
    }

    if return_episode_stats:
        stats["episodes"] = episode_details

    return stats


# ---------------------------------------------------------
# 🔥 TOP-LEVEL FUNCTION (NOT NESTED)
# ---------------------------------------------------------
def evaluate_ddqn_vs_ddqn(
    checkpoint_a: str,
    checkpoint_b: str,
    num_episodes: int = 20,
    device: str = "cpu",
    max_moves: int = 300,
    verbose: bool = False,
    return_episode_stats: bool = False,
) -> Dict[str, Any]:

    env = CheckersEnv()

    agent_a = DDQNAgent(device=device)
    agent_a.load_weights(checkpoint_a)

    agent_b = DDQNAgent(device=device)
    agent_b.load_weights(checkpoint_b)

    a_wins = b_wins = draws = 0
    total_reward_a = 0.0
    total_moves = 0
    episode_details: List[Dict[str, Any]] = []

    for ep in range(num_episodes):
        starting_player = 1 if ep % 2 == 0 else -1

        agent_first = agent_a if starting_player == 1 else agent_b
        agent_second = agent_b if starting_player == 1 else agent_a

        result = play_game(env, agent_first, agent_second, max_moves, starting_player, verbose)

        winner = result["winner"]
        if winner == result["first_player"]:
            a_wins += 1
        elif winner == result["second_player"]:
            b_wins += 1
        else:
            draws += 1

        total_reward_a += result["total_reward"] if starting_player == 1 else -result["total_reward"]
        total_moves += result["moves"]

        if return_episode_stats:
            episode_details.append(result)

    return {
        "num_episodes": num_episodes,
        "agent_a_wins": a_wins,
        "agent_b_wins": b_wins,
        "draws": draws,
        "agent_a_win_rate": a_wins / num_episodes,
        "agent_b_win_rate": b_wins / num_episodes,
        "draw_rate": draws / num_episodes,
        "avg_agent_a_reward": total_reward_a / num_episodes,
        "avg_moves": total_moves / num_episodes,
        "episodes": episode_details if return_episode_stats else None,
    }
