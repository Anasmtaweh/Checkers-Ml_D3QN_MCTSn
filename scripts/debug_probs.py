import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os
import sys
import glob

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.action_manager import ActionManager
from training.alpha_zero.network import AlphaZeroModel

def check_probability_sums():
    print("🔍 DIAGNOSTIC: Checking Policy Probability Sums...")
    
    # 1. Setup Device & Components
    device = "cuda" if torch.cuda.is_available() else "cpu"
    action_manager = ActionManager(device)
    
    # 2. Find Latest Model
    checkpoint_dir = "checkpoints/alphazero"
    list_of_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_iter_*.pth'))
    if not list_of_files:
        print("❌ No checkpoints found.")
        return
    latest_model_path = max(list_of_files, key=os.path.getctime)
    print(f"  Loading Model: {os.path.basename(latest_model_path)}")
    
    # 3. Load Model
    model = AlphaZeroModel(action_dim=action_manager.action_dim, device=device)
    try:
        checkpoint = torch.load(latest_model_path, map_location=device)
        model.network.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 4. Load Replay Buffer
    buffer_path = os.path.join(checkpoint_dir, "latest_replay_buffer.pkl")
    if not os.path.exists(buffer_path):
        print("❌ No replay buffer found (train for a bit longer or save buffer).")
        return
    
    print(f"  Loading Buffer: {buffer_path}")
    try:
        with open(buffer_path, 'rb') as f:
            replay_buffer = pickle.load(f)
    except Exception as e:
        print(f"❌ Failed to load buffer: {e}")
        return

    if len(replay_buffer) < 10:
        print("⚠️  Buffer too small to test.")
        return

    # 5. Sample & Check
    print(f"  Sampling 64 positions from buffer of size {len(replay_buffer)}...")
    indices = np.random.choice(len(replay_buffer), 64, replace=True)
    
    states_list = []
    policy_targets_list = []
    
    for idx in indices:
        state, policy, _ = replay_buffer[idx]
        states_list.append(state)
        policy_targets_list.append(policy)
    
    # Stack batch
    states = torch.stack(states_list).to(device)
    policy_targets = torch.tensor(np.array(policy_targets_list), dtype=torch.float32).to(device)
    
    # Run Inference
    with torch.no_grad():
        policy_logits, _ = model.get_policy_value(states)
        
        # MATH CHECK
        # 1. Target Sum (Should be exactly 1.0)
        target_sum = policy_targets.sum(dim=1).mean().item()
        
        # 2. Prediction Sum (exp(logits) should be ~1.0)
        pred_sum = torch.exp(policy_logits).sum(dim=1).mean().item()
        
    print("\n" + "="*50)
    print(f"📊 RESULTS")
    print("="*50)
    print(f"Target Sum (Ground Truth): {target_sum:.6f}  (Should be 1.0)")
    print(f"Pred Sum   (Model Output): {pred_sum:.6f}  (Should be ~1.0)")
    
    if 0.99 < pred_sum < 1.01:
        print("✅ PASS: Your model outputs valid probabilities.")
    else:
        print("❌ FAIL: Your model outputs are NOT valid probabilities. Check LogSoftmax.")
    print("="*50 + "\n")

if __name__ == "__main__":
    check_probability_sums()