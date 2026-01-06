import json
import matplotlib.pyplot as plt
import sys
import os
import glob

def plot_single_gen(json_file):
    if not os.path.exists(json_file):
        print(f"⚠️ Skipping {json_file} (Not found)")
        return

    # Create a fresh figure for this generation
    plt.figure(figsize=(12, 6))

    with open(json_file, "r") as f:
        data = json.load(f)

    # Sort by iteration
    data.sort(key=lambda x: x['iteration'])
    iters = [d['iteration'] for d in data]
    elos = [d['elo'] for d in data]
    
    gen_name = json_file.replace("results_", "").replace(".json", "")

    # Plot
    plt.plot(iters, elos, marker='o', linestyle='-', color='blue', linewidth=2)
    plt.fill_between(iters, elos, min(elos), color='blue', alpha=0.1)

    # Find Max
    best = max(data, key=lambda x: x['elo'])
    plt.plot(best['iteration'], best['elo'], 'r*', markersize=15)
    plt.annotate(f"CHAMPION\nIter {best['iteration']}\nElo {int(best['elo'])}", 
                 (best['iteration'], best['elo']),
                 xytext=(0, 15), textcoords='offset points',
                 ha='center', color='darkred', fontweight='bold')

    plt.title(f"Skill Progression: {gen_name}", fontsize=14)
    plt.xlabel("Training Iteration", fontsize=12)
    plt.ylabel("Elo Rating", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    output_file = f"graph_{gen_name}.png"
    plt.savefig(output_file)
    plt.close() # Close memory
    print(f"✅ Saved graph to {output_file}")

def main():
    # Automatically find all result JSONs in the current folder
    files = glob.glob("results_*.json")
    
    if not files:
        print("❌ No results_*.json files found! Run evaluate_generations.py first.")
        return

    for json_file in files:
        plot_single_gen(json_file)

if __name__ == "__main__":
    main()