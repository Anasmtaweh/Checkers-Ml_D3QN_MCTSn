"""
INTEGRATION EXAMPLE: How Phase 1 and Phase 2 Work Together
===========================================================

This example demonstrates the complete data flow from raw checkers
board state through encoding, model forward pass, and action selection.
"""

import torch
import numpy as np

# Assume we have these imports (from Phase 1 and Phase 2)
# from training.common import ActionManager, CheckersBoardEncoder
# from training.d3qn.model import D3QNModel

def integration_demo():
    """
    Complete pipeline demonstration.
    """

    print("="*70)
    print("INTEGRATION EXAMPLE: Board → Encoding → Model → Action")
    print("="*70)

    # ====================================================================
    # STEP 1: Initialize Components
    # ====================================================================
    print("\nStep 1: Initialize Components")
    print("-"*70)

    # Initialize action manager (creates universal action space)
    action_manager = ActionManager(device="cpu")
    print(f"✓ ActionManager initialized with {action_manager.action_dim} actions")

    # Initialize board encoder
    encoder = CheckersBoardEncoder()
    print(f"✓ BoardEncoder initialized")

    # Initialize D3QN model
    model = D3QNModel(action_dim=action_manager.action_dim, device="cpu")
    print(f"✓ D3QNModel initialized with {count_parameters(model.online):,} parameters")

    # ====================================================================
    # STEP 2: Create Sample Board State
    # ====================================================================
    print("\nStep 2: Sample Board State")
    print("-"*70)

    # Create a mid-game board state
    board = np.zeros((8, 8), dtype=np.int32)

    # Player 1 pieces (positive values)
    board[2, 1] = 1   # Regular man
    board[3, 4] = 2   # King
    board[4, 3] = 1   # Regular man

    # Player -1 pieces (negative values)
    board[5, 2] = -1  # Regular man
    board[6, 5] = -2  # King
    board[7, 4] = -1  # Regular man

    current_player = 1

    print("Raw board state:")
    print(board)
    print(f"Current player: {current_player}")

    # ====================================================================
    # STEP 3: Encode Board State
    # ====================================================================
    print("\nStep 3: Encode Board State")
    print("-"*70)

    # Encode the board (canonicalization happens here)
    encoded_state = encoder.encode(board, player=current_player, info=None)

    print(f"Encoded state shape: {encoded_state.shape}")
    print(f"  Channel 0 (My men):     {encoded_state[0].sum():.0f} pieces")
    print(f"  Channel 1 (My kings):   {encoded_state[1].sum():.0f} pieces")
    print(f"  Channel 2 (Enemy men):  {encoded_state[2].sum():.0f} pieces")
    print(f"  Channel 3 (Enemy kings): {encoded_state[3].sum():.0f} pieces")
    print(f"  Channel 4 (Tempo):      {encoded_state[4][0,0]:.1f} (0=P1, 1=P2)")

    # ====================================================================
    # STEP 4: Get Legal Moves from Environment
    # ====================================================================
    print("\nStep 4: Legal Moves (Simulated)")
    print("-"*70)

    # Simulate legal moves from environment
    legal_moves = [
        ((2, 1), (3, 2)),  # Simple move
        ((4, 3), (5, 4)),  # Simple move
        [((3, 4), (5, 2), (4, 3))],  # Capture sequence
    ]

    print(f"Number of legal moves: {len(legal_moves)}")
    for i, move in enumerate(legal_moves):
        print(f"  Move {i+1}: {move}")

    # ====================================================================
    # STEP 5: Create Legal Action Mask
    # ====================================================================
    print("\nStep 5: Create Legal Action Mask")
    print("-"*70)

    mask = action_manager.make_legal_action_mask(legal_moves)

    print(f"Mask shape: {mask.shape}")
    print(f"Number of legal actions: {mask.sum().item()}")
    print(f"Legal action indices: {torch.where(mask)[0].tolist()}")

    # ====================================================================
    # STEP 6: Forward Pass Through Model
    # ====================================================================
    print("\nStep 6: Model Forward Pass")
    print("-"*70)

    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        q_values = model.get_q_values(encoded_state)

    print(f"Q-values shape: {q_values.shape}")
    print(f"Q-values range: [{q_values.min().item():.4f}, {q_values.max().item():.4f}]")
    print(f"Q-values mean: {q_values.mean().item():.4f}")

    # ====================================================================
    # STEP 7: Apply Legal Move Mask
    # ====================================================================
    print("\nStep 7: Apply Legal Move Mask")
    print("-"*70)

    masked_q_values = q_values.clone()
    masked_q_values[~mask] = -1e9  # Set illegal actions to very negative

    legal_q_values = q_values[mask]
    print(f"Legal Q-values range: [{legal_q_values.min().item():.4f}, "
          f"{legal_q_values.max().item():.4f}]")

    # ====================================================================
    # STEP 8: Select Best Action
    # ====================================================================
    print("\nStep 8: Action Selection")
    print("-"*70)

    best_action_id = torch.argmax(masked_q_values).item()
    best_move = action_manager.get_move_from_id(best_action_id)
    best_q_value = q_values[0, best_action_id].item()

    print(f"Best action ID: {best_action_id}")
    print(f"Best move: {best_move}")
    print(f"Q-value: {best_q_value:.4f}")

    # Map back to original environment move format
    # (In practice, use parse_legal_moves to get the mapping)
    print(f"\nThis (start, landing) pair would be mapped back to the")
    print(f"original environment move format before passing to env.step()")

    # ====================================================================
    # STEP 9: Alternative - Epsilon-Greedy Exploration
    # ====================================================================
    print("\nStep 9: Epsilon-Greedy Action Selection")
    print("-"*70)

    epsilon = 0.1  # 10% random actions

    if np.random.random() < epsilon:
        # Random action from legal moves
        legal_action_ids = torch.where(mask)[0]
        random_action_id = legal_action_ids[np.random.randint(len(legal_action_ids))]
        selected_action_id = random_action_id.item()
        selection_type = "RANDOM (exploration)"
    else:
        # Greedy action
        selected_action_id = best_action_id
        selection_type = "GREEDY (exploitation)"

    selected_move = action_manager.get_move_from_id(selected_action_id)
    selected_q_value = q_values[0, selected_action_id].item()

    print(f"Selection type: {selection_type}")
    print(f"Selected action ID: {selected_action_id}")
    print(f"Selected move: {selected_move}")
    print(f"Q-value: {selected_q_value:.4f}")

    # ====================================================================
    # STEP 10: Demonstrate Canonicalization
    # ====================================================================
    print("\nStep 10: Canonicalization Demo (Player -1)")
    print("-"*70)

    # Encode same board from Player -1's perspective
    encoded_state_p2 = encoder.encode(board, player=-1, info=None)

    print(f"Player -1 encoding:")
    print(f"  Channel 0 (My men):     {encoded_state_p2[0].sum():.0f} pieces")
    print(f"  Channel 1 (My kings):   {encoded_state_p2[1].sum():.0f} pieces")
    print(f"  Channel 2 (Enemy men):  {encoded_state_p2[2].sum():.0f} pieces")
    print(f"  Channel 3 (Enemy kings): {encoded_state_p2[3].sum():.0f} pieces")
    print(f"  Channel 4 (Tempo):      {encoded_state_p2[4][0,0]:.1f} (0=P1, 1=P2)")

    print(f"\nNote: Board is rotated 180° and piece IDs swapped")
    print(f"      so Player -1 always sees themselves as 'player 1'")

    # Get Q-values for Player -1's perspective
    with torch.no_grad():
        q_values_p2 = model.get_q_values(encoded_state_p2)

    print(f"\nQ-values will be different due to canonicalization,")
    print(f"but the model uses the same learned strategy!")

    print("\n" + "="*70)
    print("INTEGRATION COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. ActionManager maps all moves to fixed indices")
    print("  2. BoardEncoder creates consistent 5-channel representation")
    print("  3. Model outputs Q-values for all possible actions")
    print("  4. Legal move mask ensures only valid actions selected")
    print("  5. Canonicalization enables symmetric learning")
    print("="*70)


# ====================================================================
# TRAINING PIPELINE PREVIEW
# ====================================================================

def training_step_preview():
    """
    Preview of what a single training step will look like.
    """

    print("\n" + "="*70)
    print("TRAINING STEP PREVIEW (Phase 3)")
    print("="*70)

    print("""
    while training:
        # 1. Agent plays a move (epsilon-greedy)
        state = encoder.encode(board, player)
        q_values = model.get_q_values(state)
        masked_q = apply_legal_mask(q_values, legal_moves)
        action = select_action(masked_q, epsilon)

        # 2. Environment responds
        next_state, reward, done, info = env.step(action)

        # 3. Store transition in replay buffer
        replay_buffer.add(state, action, reward, next_state, done)

        # 4. Sample batch and compute loss
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)

            # Current Q-values
            q_current = model.online(batch.states)
            q_current = q_current.gather(1, batch.actions)

            # Target Q-values (Double Q-learning)
            with torch.no_grad():
                # Online network selects actions
                q_next_online = model.online(batch.next_states)
                best_actions = q_next_online.argmax(dim=1)

                # Target network evaluates those actions
                q_next_target = model.target(batch.next_states)
                q_next = q_next_target.gather(1, best_actions.unsqueeze(1))

                # TD target
                q_target = batch.rewards + gamma * q_next * (1 - batch.dones)

            # Compute loss (Huber or MSE)
            loss = F.smooth_l1_loss(q_current, q_target)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.online.parameters(), 1.0)
            optimizer.step()

        # 5. Update target network periodically
        if step % target_update_freq == 0:
            model.update_target_network()

        # 6. Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    """)

    print("="*70)


if __name__ == "__main__":
    # Note: This would run if you have all dependencies
    # For now, it's a template showing the complete pipeline

    print("This is an integration template.")
    print("Run this after implementing all components to verify the pipeline.")
    print("\nData Flow:")
    print("  Raw Board (8x8 numpy)")
    print("    ↓ [BoardEncoder]")
    print("  Encoded State (5, 8, 8 tensor)")
    print("    ↓ [D3QNModel]")
    print("  Q-Values (action_dim tensor)")
    print("    ↓ [Legal Mask + argmax]")
    print("  Selected Action (integer)")
    print("    ↓ [ActionManager.id_to_move]")
    print("  Environment Move (tuple)")
    print("    ↓ [env.step]")
    print("  Next State + Reward")
