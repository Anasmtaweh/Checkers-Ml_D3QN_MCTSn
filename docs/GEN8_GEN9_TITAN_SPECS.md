# Gen 8 and Gen 9 Titan Specifications

## Gen 8 Documentation

### Identity
*   **Name:** `gen8_titan`
*   **Origin:** D3QN checkpoint around episode 2500 (`iron_nuclear_2500.pth`)
*   **Role:** Previous main champion before Iron League Phase 5.
*   **Status:** Retired from "champion" role, kept in the opponent pool as a strong baseline.

### Strength
*   **Tournament performance** (Swiss, 6 rounds, 10 games per pairing):
    *   **Final leaderboard rank:** 4th.
    *   **Tournament win rate:** 58.3%.
*   **Typical vs-random strength:** ~90% win rate (used as a calibration point for later gens).

### Usage
*   **Used as:**
    *   Evaluation baseline for later generations in Iron League tournaments.
    *   Strong fixed opponent in self-play opponent pools (`gen8_titan.pth`, `gen8_mirror.pth`).
*   **Architecture:** Dueling D3QN with ~170 discrete action IDs (`ActionManager`) over the checkers move space.

---

## Gen 9 Documentation

### Identity
*   **Main champion:** `gen9_titan`
    *   **File:** `iron_nuclear_4500.pth` (often aliased as `gen9_titan.pth`).
    *   **Training point:** Episode 4500 in the Iron League Phase 5 run with nuclear LR schedule and Q-health monitor.
*   **Secondary champion:** `gen9_champion`
    *   **File:** `iron_nuclear_4000.pth` (often aliased as `gen9_champion.pth`).
    *   **Training point:** Episode 4000.

### Strength
*   **Tournament** (13 competitors: all Iron League checkpoints + Gen 7/8 pool):
    *   **iron_nuclear_4500:**
        *   **Rank:** 1st of 13.
        *   **Points:** 40.0, Buchholz 197.5.
        *   **Tournament win rate:** 66.7%.
        *   **Head-to-head vs iron_nuclear_4000:** 7.5 – 2.5 (dominant).
        *   **Head-to-head vs gen8_titan:** 7.5 – 2.5.
    *   **iron_nuclear_4000:**
        *   **Rank:** 2nd of 13.
        *   **Points:** 37.5, Buchholz 197.5.
        *   **Tournament win rate:** 62.5%.
*   **External evaluation:**
    *   `iron_nuclear_4500` vs random (100 games, agent as Red, silent mode):
        *   **Wins:** 85 (85.0%)
        *   **Losses:** 15
        *   **Draws:** 0

### Training Setup (Gen 9 Run)
*   **Environment:** Custom `CheckersEnv` with `ActionManager` and 170-action discrete space.
*   **Algorithm:** Dueling D3QN with:
    *   **Replay buffer:** 40,000 transitions.
    *   **Min buffer to train:** 5,000 transitions.
    *   **Batch size:** 128.
    *   **Discount factor:** γ = 0.99.
    *   **Target soft update:** τ = 0.001.
    *   **Gradient clipping:** 0.1.
*   **Epsilon schedule:** Start: 0.30, end: 0.05, decay: 5000 episodes.
*   **Learning rate schedule** (Iron League Phase 5):
    *   Episode < 1000: 2e-6
    *   1000 ≤ episode < 3000: 1e-6
    *   3000 ≤ episode < 6000: 5e-7
    *   ≥ 6000: 2e-7
*   **Opponent mixture:**
    *   **Early:** Higher random + historical pool.
    *   **Later:** More pool + self-play; opponent pool includes Gen 7, Gen 8, and later Gen 9 checkpoints.

### Q-Health and Safety
*   **Q-health monitor:** Periodic sampling (every 50 episodes) of buffer states to log Max Q and Avg Q.
*   **Auto-brake:** When Max Q exceeded a threshold (~28–30), the LR was halved and frozen via `auto_brake_active`.
*   **Emergency stop:** Training halted when Max Q > 50 or Avg Q too large (e.g., at Episode 4700, Max Q ~50.88) to prevent full divergence.

### Role and Pool Integration
*   **gen9_titan:** Current top agent for deployment and web front-end. Added to `opponent_pool` as `gen9_titan.pth`.
*   **gen9_champion:** Backup elite agent, slightly more conservative than 4500. Added to `opponent_pool` as `gen9_champion.pth`.
*   **Gen 8 agents:** (`gen8_titan`, `gen8_mirror`) kept in pool to preserve curriculum and diversity.

### Usage Guide
*   **For documentation/Wiki:**
    *   Treat Gen 8 as "baseline D3QN champion at episode 2500".
    *   Treat Gen 9 as "Iron League D3QN champions at episodes 4000 and 4500, with 4500 as the official best model".
*   **For MCTS+NN Phase:** Refer to Gen 9 as the benchmark to beat in Tournament WR (≥ 66.7%) and vs-random (≥ 85%).
