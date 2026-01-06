import os
import glob
import torch
import shutil
import json
import argparse
import numpy as np
import ray
import sys
import time
from itertools import combinations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..')))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

# CONFIG
SIMULATIONS = 800      
GAMES_PER_MATCH = 2    
NUM_WORKERS = 8        
WINDOW_SIZE = 3 

@ray.remote(num_gpus=0.12) 
class ArenaWorker:
    def __init__(self, device="cuda"):
        import torch
        self.device = device
        torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"

    def play_match(self, p1_path, p2_path, p1_name, p2_name):
        # PRINT START HEARTBEAT
        print(f"   ... Worker starting: {p1_name} vs {p2_name}")
        
        from mcts_workspace.core.game import CheckersEnv
        from mcts_workspace.core.action_manager import ActionManager
        from mcts_workspace.core.board_encoder import CheckersBoardEncoder
        from mcts_workspace.training.alpha_zero.network import AlphaZeroModel
        from mcts_workspace.training.alpha_zero.mcts import MCTS

        try:
            manager = ActionManager(self.device)
            encoder = CheckersBoardEncoder()
            
            model1 = AlphaZeroModel(manager.action_dim, self.device)
            model1.network.load_state_dict(torch.load(p1_path, map_location=self.device)["model_state_dict"])
            model1.eval()

            model2 = AlphaZeroModel(manager.action_dim, self.device)
            model2.network.load_state_dict(torch.load(p2_path, map_location=self.device)["model_state_dict"])
            model2.eval()
        except Exception as e:
            return None, None, f"Error: {e}", 0

        score1, score2 = 0, 0
        
        for game_idx in range(GAMES_PER_MATCH):
            env = CheckersEnv(max_moves=150, no_progress_limit=40)
            env.reset()
            
            red_model = model1 if game_idx == 0 else model2
            blk_model = model2 if game_idx == 0 else model1
            
            mcts_red = MCTS(red_model, manager, encoder, num_simulations=SIMULATIONS, c_puct=1.0, dirichlet_alpha=0.0, device=self.device)
            mcts_blk = MCTS(blk_model, manager, encoder, num_simulations=SIMULATIONS, c_puct=1.0, dirichlet_alpha=0.0, device=self.device)
            
            while not env.done:
                active_mcts = mcts_red if env.current_player == 1 else mcts_blk
                with torch.no_grad():
                    probs, _ = active_mcts.get_action_prob(env, temp=0.0, training=False)
                move = active_mcts._get_move_from_action(int(np.argmax(probs)), env.get_legal_moves(), env.current_player)
                if not move: env.done = True; env.winner = -env.current_player; break
                env.step(move)

            if env.winner == 1: 
                if game_idx == 0: score1 += 1
                else: score2 += 1
            elif env.winner == -1: 
                if game_idx == 0: score2 += 1
                else: score1 += 1
            else:
                score1 += 0.5
                score2 += 0.5
        
        return p1_name, score1, p2_name, score2

def main():
    parser = argparse.ArgumentParser()
    default_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'alphazero')
    parser.add_argument("--folder", type=str, default=default_path)
    parser.add_argument("--name", type=str, default="Final_Parallel_Opt")
    parser.add_argument("--step", type=int, default=5)
    args = parser.parse_args()

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    print(f"🚀 INITIALIZING OPTIMIZED PARALLEL TOURNAMENT")
    
    if not os.path.exists(args.folder):
        print(f"❌ Error: Folder not found: {args.folder}")
        return

    all_files = sorted(glob.glob(os.path.join(args.folder, "*.pth")), 
                   key=lambda x: int(x.split("_")[-1].replace(".pth", "")))
    
    CRITICAL_ITERS = [142, 155, 167, 172, 181, 185, 189, 190, 194, 198, 200]
    
    players = [] 
    for f in all_files:
        try:
            iter_num = int(f.split("_")[-1].replace(".pth", ""))
            if (iter_num % args.step == 0) or (iter_num in CRITICAL_ITERS):
                players.append({
                    'path': f, 'iter': iter_num, 'name': f"Iter_{iter_num}", 'elo': 1000
                })
        except: pass

    print(f"✅ Contestants: {len(players)} models.")

    # --- GENERATE MATCHUPS ---
    matchups = []
    added_pairs = set()
    players.sort(key=lambda x: x['iter'])
    crit_indices = [i for i, p in enumerate(players) if p['iter'] in CRITICAL_ITERS]

    for i in range(len(players)):
        start_j = max(0, i - WINDOW_SIZE)
        end_j = min(len(players), i + WINDOW_SIZE + 1)
        for j in range(start_j, end_j):
            if i == j: continue
            pair = tuple(sorted((i, j)))
            if pair not in added_pairs:
                matchups.append((i, j))
                added_pairs.add(pair)
        for c_idx in crit_indices:
            if i == c_idx: continue
            pair = tuple(sorted((i, c_idx)))
            if pair not in added_pairs:
                matchups.append((i, c_idx))
                added_pairs.add(pair)

    print(f"🥊 Total Matches: {len(matchups)}")

    workers = [ArenaWorker.remote() for _ in range(NUM_WORKERS)]
    futures = []
    
    # --- QUEUE SYSTEM: ONLY FEED 4 AT A TIME ---
    pending_matchups = matchups.copy()
    completed = 0
    results_log = [] 
    start_time = time.time()
    
    # Initial fill
    for _ in range(min(NUM_WORKERS, len(pending_matchups))):
        idx1, idx2 = pending_matchups.pop(0)
        p1 = players[idx1]
        p2 = players[idx2]
        # Pick a worker in round-robin or random, doesn't matter since they are stateless
        futures.append(workers[len(futures)].play_match.remote(p1['path'], p2['path'], p1['name'], p2['name']))  # type: ignore

    print("⏳ Running matches...")
    print("-" * 60)
    
    while futures:
        # Wait for at least one to finish
        done_id, futures = ray.wait(futures, num_returns=1)
        p1_name, s1, p2_name, s2 = ray.get(done_id[0])
        
        completed += 1
        elapsed = time.time() - start_time
        rate = completed / max(1, elapsed) 
        eta_min = ((len(matchups) - completed) / rate) / 60 if rate > 0 else 0
        
        status = "DRAW" if s1 == s2 else (p1_name if s1 > s2 else p2_name)
        print(f"[{completed}/{len(matchups)}] {p1_name} vs {p2_name}: {s1}-{s2} ({status}) | ETA: {eta_min:.1f}m", flush=True)
        
        if isinstance(p1_name, str): 
            results_log.append((p1_name, s1, p2_name, s2))

        # Feed next match if available
        if pending_matchups:
            idx1, idx2 = pending_matchups.pop(0)
            p1 = players[idx1]
            p2 = players[idx2]
            # Add to the worker slot that just freed up? 
            # Ray manages the queue, we just throw it in.
            # Using specific worker index is tricky here, so we just use round robin on the worker pool handle
            worker_idx = completed % NUM_WORKERS
            futures.append(workers[worker_idx].play_match.remote(p1['path'], p2['path'], p1['name'], p2['name']))  # type: ignore

        if completed % 10 == 0:
            with open(os.path.join(PROJECT_ROOT, "partial_results_backup.json"), "w") as f:
                json.dump(results_log, f)

    print("-" * 60)
    print("\n✅ All matches complete. Calculating Elo...")

    player_stats = {p['name']: {'elo': 1000, 'wins': 0, 'losses': 0, 'draws': 0, 'iter': p['iter'], 'path': p['path']} for p in players}
    
    for p1n, s1, p2n, s2 in results_log:
        K = 32
        p1 = player_stats[p1n]
        p2 = player_stats[p2n]
        e1 = 1 / (1 + 10 ** ((p2['elo'] - p1['elo']) / 400))
        e2 = 1 / (1 + 10 ** ((p1['elo'] - p2['elo']) / 400))
        p1['elo'] += K * (s1 - (e1 * GAMES_PER_MATCH))
        p2['elo'] += K * (s2 - (e2 * GAMES_PER_MATCH))
        if s1 > s2: p1['wins']+=1; p2['losses']+=1
        elif s2 > s1: p2['wins']+=1; p1['losses']+=1
        else: p1['draws']+=1; p2['draws']+=1

    sorted_players = sorted(player_stats.values(), key=lambda x: x['elo'], reverse=True)
    output_dir = os.path.join(PROJECT_ROOT, "gen_champions") 
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n🏆 FINAL LEADERBOARD:")
    export_data = []
    
    for i, p in enumerate(sorted_players):
        print(f"   #{i+1} | {p['name']} | Elo: {int(p['elo'])} | W-L-D: {p['wins']}-{p['losses']}-{p['draws']}")
        export_data.append({"iteration": p['iter'], "elo": p['elo'], "name": p['name']})
        if i < 3:
            dest = os.path.join(output_dir, f"{i+1}_{args.name}.pth")
            shutil.copy(p['path'], dest)

    with open(os.path.join(PROJECT_ROOT, f"results_{args.name}.json"), "w") as f:
        json.dump(export_data, f, indent=4)
    ray.shutdown()

if __name__ == "__main__":
    main()