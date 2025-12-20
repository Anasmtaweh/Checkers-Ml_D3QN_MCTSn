import torch

print("\n" + "="*70)
print("CHECKPOINT DIAGNOSTICS")
print("="*70)

checkpoints_to_check = [
    "checkpoints_iron_league/iron_agent_800.pth",
    "checkpoints_iron_league/iron_agent_1000.pth", 
    "opponent_pool/gen7_final.pth"
]

for ckpt_path in checkpoints_to_check:
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        print(f"\n{ckpt_path}:")
        
        if isinstance(checkpoint, dict):
            print(f"  Type: Full checkpoint dict")
            print(f"  Keys: {list(checkpoint.keys())}")
            
            if 'model_online' in checkpoint:
                weights = checkpoint['model_online']
            else:
                weights = checkpoint
        else:
            print(f"  Type: Raw state_dict (weights only)")
            weights = checkpoint
        
        sample_weight = next(iter(weights.values()))
        print(f"  Weight dtype: {sample_weight.dtype}")
        print(f"  Weight stats: min={sample_weight.min():.4f}, max={sample_weight.max():.4f}")
        
        # Check for NaN/Inf
        has_nan = any(torch.isnan(v).any() for v in weights.values())
        has_inf = any(torch.isinf(v).any() for v in weights.values())
        print(f"  Has NaN: {has_nan}")
        print(f"  Has Inf: {has_inf}")
        
        if has_nan or has_inf:
            print(f"  ❌ CORRUPTED! Contains NaN/Inf values")
        else:
            print(f"  ✅ Weights look valid")
            
    except FileNotFoundError:
        print(f"\n{ckpt_path}:")
        print(f"  ❌ FILE NOT FOUND")
    except Exception as e:
        print(f"\n{ckpt_path}:")
        print(f"  ❌ ERROR: {e}")

print("\n" + "="*70)