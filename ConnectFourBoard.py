"""
ConnectFour Initialization
"""

from re import L


class ConnectFourBoard:

    def __init__(self, height=6, width=7):
        self.height = height
        self.width = width
        self.board = [['0' for x in range(width)] for i in range(height)]

    def get_column(self, index):
        return [i[index] for i in self.board]

    def get_row(self, index):
        return self.board[index]

    def is_on_board(self, row, col):
        return row < self.height and col < self.width

    def get_space(self, row, col):
        return self.board[row][col]

    def get_diagonals(self):
        diagonals = []

        for i in range(self.height + self.width - 1):
            diagonals.append([])
            for j in range(max(i - self.height + 1, 0), min(i + 1, self.height)):
                diagonals[i].append(self.board[self.height - i + j - 1][j])

        for i in range(self.height + self.width - 1):
            diagonals.append([])
            for j in range(max(i - self.height + 1, 0), min(i + 1, self.height)):
                diagonals[i].append(self.board[i - j][j])

        return diagonals

    def available(self, row, col):
        return self.get_space(row, col) == 0

    def filled(self, row, col):
        return not self.get_space(row, col) == 0

    def is_win(self, spaces):
        # Check if all of the spaces have the same value
        last_space = None
        for space in spaces:
            row, col = space
            if self.is_on_board(row, col):
                space_value = self.get_space(row, col)
                # If any of the spaces are empty then no one could win
                if (space_value == 0): return 0
                if (last_space != None and last_space != space_value):
                    return 0
                else:
                    last_space = space_value
            else:
                return 0
        return last_space

    def someone_wins(self, move):
        # Get the row and col of the move
        row, col = move
        # Check up and down
        up = [(row + x, col) for x in range(1,5)]
        up_win = self.is_win(up)
        if up_win != 0:
            return up_win
        down = [(row - x, col) for x in range(1,5)]
        down_win = self.is_win(down)
        if down_win != 0:
            return down_win
        # Check left and right
        right = [(row, col + x) for x in range(1,5)]
        right_win = self.is_win(right)
        if right_win != 0:
            return right_win
        left = [(row, col - x) for x in range(1,5)]
        left_win = self.is_win(left)
        if left_win != 0:
            return left_win
        # Check diagonals
        diagonals = [
            [(row - x, col - x) for x in range(1,5)],
            [(row - x, col + x) for x in range(1,5)],
            [(row + x, col + x) for x in range(1,5)],
            [(row + x, col - x) for x in range(1,5)]
        ]
        for diagonal in diagonals:
            diagonal_win = self.is_win(diagonal)
            if diagonal_win != 0:
                return diagonal_win
        # If someone hasn't won after all these checks then no one has won
        return 0

if __name__ == '__main__':
    # start_game()
    board = ConnectFourBoard()
    #print(game.board)
    print(board.get_diagonals())
