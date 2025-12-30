from core.board_encoder import CheckersBoardEncoder
import numpy as np

encoder = CheckersBoardEncoder()

# Create simple test position
board = np.zeros((8, 8), dtype=int)
board[0, 0] = 2  # P2 piece
board[7, 7] = 1  # P1 piece

# Encode from P1's perspective
encoded_p1 = encoder.encode(board, 1)
print("Encoded as P1's turn:")
print(f"  Shape: {encoded_p1.shape}")
print(f"  Channel 0 (should be MY pieces): {encoded_p1[0].sum()}")
print(f"  Channel 2 (should be OPP pieces): {encoded_p1[2].sum()}")

# Encode from P2's perspective  
encoded_p2 = encoder.encode(board, 2)
print("\nEncoded as P2's turn:")
print(f"  Shape: {encoded_p2.shape}")
print(f"  Channel 0 (should be MY pieces): {encoded_p2[0].sum()}")
print(f"  Channel 2 (should be OPP pieces): {encoded_p2[2].sum()}")

# Test symmetry - When P1 plays, P1 pieces should be in channel 0
# When P2 plays, P2 pieces should ALSO be in channel 0 (canonicalized)
# Original board: P1 piece at (7,7), P2 piece at (0,0)

print("\n" + "="*60)
print("SYMMETRY TEST:")
print("="*60)

# When P1 to move: Channel 0 should have P1's piece (at 7,7)
p1_my_pieces = encoded_p1[0].sum().item()
# When P2 to move: Channel 0 should have P2's piece (rotated from 0,0 to 7,7)
p2_my_pieces = encoded_p2[0].sum().item()

print(f"P1 to move - MY pieces count: {p1_my_pieces}")
print(f"P2 to move - MY pieces count: {p2_my_pieces}")

if p1_my_pieces == 1.0 and p2_my_pieces == 1.0:
    print("\n✅ Encoder is SYMMETRIC (correct!)")
    print("   Both players see 1 of their own pieces in channel 0")
else:
    print("\n❌ Encoder is BROKEN!")
    print(f"   Expected both to be 1.0, got {p1_my_pieces} and {p2_my_pieces}")

# Additional verification
print("\n" + "="*60)
print("DETAILED CHANNEL BREAKDOWN:")
print("="*60)
print("P1 to move:")
for i in range(5):
    print(f"  Channel {i}: {encoded_p1[i].sum().item()}")

print("\nP2 to move:")
for i in range(5):
    print(f"  Channel {i}: {encoded_p2[i].sum().item()}")
