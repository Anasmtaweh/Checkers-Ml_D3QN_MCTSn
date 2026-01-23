import numpy as np


class CheckersBoard:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((8, 8), dtype=int)

        # Player 1 = 1
        # Player -1 = -1
        # Kings = 2 / -2

        # Player 1 starts at top 3 rows
        for r in range(3):
            for c in range(8):
                if (r + c) % 2 == 1:
                    self.board[r][c] = 1

        # Player -1 bottom 3 rows
        for r in range(5, 8):
            for c in range(8):
                if (r + c) % 2 == 1:
                    self.board[r][c] = -1

        return self.board

    def get_state(self):
        return self.board.copy()

    def move_piece(self, r1, c1, r2, c2):
        piece = self.board[r1, c1]
        self.board[r1, c1] = 0
        self.board[r2, c2] = piece

        # Promotion
        if piece == 1 and r2 == 7:
            self.board[r2, c2] = 2
        elif piece == -1 and r2 == 0:
            self.board[r2, c2] = -2

    def print_board(self):
        print("\nCurrent Board:")
        print(self.board)
