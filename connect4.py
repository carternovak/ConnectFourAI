"""
ConnectFour Initialization
"""

class ConnectFour:

    def __init__(self, height=6, width=7):
        self.height = height
        self.width = width
        self.board = [['0' for x in range(width)] for i in range(height)]

    def get_column(self, index):
        return [i[index] for i in self.board]

    def get_row(self, index):
        return self.board[index]

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

if __name__ == '__main__':
    # start_game()
    game = ConnectFour()
    #print(game.board)
    print(game.get_diagonals())