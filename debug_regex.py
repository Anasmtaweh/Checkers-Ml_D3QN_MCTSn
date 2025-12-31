
import re
import os

trainer_file = 'training/alpha_zero/trainer.py'
with open(trainer_file, 'r') as f:
    trainer_source = f.read()

print(f"File length: {len(trainer_source)}")

# Check 3.4 logic
match = re.search(r'def play_game_remote\(.*?\):(.*?)env = CheckersEnv', trainer_source, re.DOTALL)
if not match:
    print("❌ Could not extract play_game_remote body")
else:
    worker_body = match.group(1)
    print(f"Captured body length: {len(worker_body)}")
    print("-" * 20)
    print(worker_body)
    print("-" * 20)
    
    if 'mcts = MCTS(' not in worker_body:
        print("❌ MCTS construction not found in play_game_remote body")
    else:
        print("✅ MCTS construction found")

    if 'search_draw_bias=params["search_draw_bias"]' not in worker_body:
        print("❌ search_draw_bias not passed to MCTS constructor")
    else:
         print("✅ search_draw_bias found")
