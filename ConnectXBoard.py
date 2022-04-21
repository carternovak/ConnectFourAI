"""
ConnectFour Initialization
"""


class ConnectXBoard:

    # X represents the number of pieces a player needs to get in a row to win (4 for regular Connect 4)
    def __init__(self, height=6, width=7, x=4, existing_board = None):
        self.height = height
        self.width = width
        self.x = x
        if (existing_board):
            self.board = [[existing_board.get_spot(row, col) for col in range(width)] for row in range(height)]
        else:
            self.board = [['0' for x in range(width)] for i in range(height)]
        if (existing_board):
            self.first_empty_in_col = [first_empty for first_empty in existing_board.first_empty_in_col]
        else:
            self.first_empty_in_col = [0 for col in range(width)]
        self.winner = None

    def get_column(self, index):
        return [i[index] for i in self.board]

    def get_row(self, index):
        return self.board[index]

    def is_on_board(self, row, col):
        return row < self.height and col < self.width

    def get_space(self, row, col):
        return self.board[row][col]

    # Return a deep copy of the current board
    def clone_board(self):
        return ConnectXBoard(self.height, self.width, self.x, self)

    # Returns a deep copy of the current board with all the spots set to 0
    def clone_empty(self):
        return ConnectXBoard(self.height, self.width, self.x)

    def set_space(self, row, col, new_val):
        if self.get_empty_index_in_col(col) == row: 
            self.first_empty_in_col[col] += 1;
        self.board[row][col] = new_val

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

    def col_available(self, col):
       return self.first_empty_in_col[col] < self.height
       #return self.available(0, col)

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

    def check_for_win(self, move):
        if len(self.get_available_moves()) <= 0: return 0
        # Get the row and col of the move
        row, col = move
        # Check up and down
        up = [(row + x, col) for x in range(1,self.x)]
        up_win = self.is_win(up)
        if up_win != 0:
            return up_win
        down = [(row - x, col) for x in range(1,self.x)]
        down_win = self.is_win(down)
        if down_win != 0:
            return down_win
        # Check left and right
        right = [(row, col + x) for x in range(1,self.x)]
        right_win = self.is_win(right)
        if right_win != 0:
            return right_win
        left = [(row, col - x) for x in range(1,self.x)]
        left_win = self.is_win(left)
        if left_win != 0:
            return left_win
        # Check diagonals
        diagonals = [
            [(row - x, col - x) for x in range(1,self.x)],
            [(row - x, col + x) for x in range(1,self.x)],
            [(row + x, col + x) for x in range(1,self.x)],
            [(row + x, col - x) for x in range(1,self.x)]
        ]
        for diagonal in diagonals:
            diagonal_win = self.is_win(diagonal)
            if diagonal_win != 0:
                return diagonal_win
        # If someone hasn't won after all these checks then no one has won
        return None

        # Gets the columns that can have a piece placed in them
    def get_available_moves(self):
        return [col for col in range(self.width) if self.col_available(col)]

    def get_empty_index_in_col(self, col):
        return self.first_empty_in_col[col]
        #for index, val in enumerate(self.get_row(row)):
        #    if (val != 0):
        #        return index - 1

    # Places a piece in the given row, doesn't mutate the board but instead returns a new board
    def make_move(self, col, player):
        if not self.col_available(col):
            raise RuntimeError("Column is full")
        # Find the first empty row in the column
        row_index = self.get_empty_index_in_col(self, col)
        new_board = self.clone()
        new_board = new_board.set_space(row_index, col, player)
        # Check if anyone won
        win = new_board.check_for_win(row_index, col)
        # If someone won set them as the winner
        if (win != None):
            new_board.winner = win
        return new_board


    def get_board_as_array(self):
        return self.board

    def get_board_as_vector(self):
        # Flatten the board into a vector (1D array)
        board_vector = [item for row in self.get_board_as_array() for item in row]
        return board_vector

if __name__ == '__main__':
    # start_game()
    board = ConnectXBoard()
    #print(game.board)
    print(board.get_diagonals())
