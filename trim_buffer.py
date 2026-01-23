import pickle
import os
from collections import deque

# CONFIGURATION FOR YOUR PLAN
BUFFER_PATH = "checkpoints/alphazero/latest_replay_buffer.pkl"
KEEP_COUNT = 20000     # Keeps exactly Iter 142 -> 155 (The Awakening Era)
NEW_MAX_LEN = 50000    # Allows buffer to grow 2.5x larger before forgetting again

def surgical_trim():
    if not os.path.exists(BUFFER_PATH):
        print(f"❌ Error: No buffer found at {BUFFER_PATH}")
        return

    print(f"📂 Loading Replay Buffer... (75,000 samples)")
    with open(BUFFER_PATH, "rb") as f:
        buffer = pickle.load(f)

    current_size = len(buffer)
    print(f"   Current Size: {current_size} samples")

    # THE SURGERY
    print(f"🔪 Cutting the 'Dark Ages' (Iter 1-141)...")
    print(f"   Keeping latest {KEEP_COUNT} samples (Iter 142-155)...")
    
    new_data = list(buffer)[-KEEP_COUNT:]
    
    # Create new deque with your requested 50,000 limit
    new_buffer = deque(new_data, maxlen=NEW_MAX_LEN)

    print(f"   New Size: {len(new_buffer)} samples")
    print(f"   New Capacity: {new_buffer.maxlen} samples")

    print(f"💾 Saving optimized buffer...")
    with open(BUFFER_PATH, "wb") as f:
        pickle.dump(new_buffer, f)
    
    print("✅ Surgery Complete.")

if __name__ == "__main__":
    surgical_trim()