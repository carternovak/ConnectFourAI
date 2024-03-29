"""
ConnectFour Initialization
"""
import torch

class ConnectXBoard:

    # X represents the number of pieces a player needs to get in a row to win (4 for regular Connect 4)
    def __init__(self, height=6, width=7, x=4, existing_board = None):
        self.height = height
        self.width = width
        self.x = x

        if (existing_board):
            self.board = [[existing_board.get_space(row, col) for col in range(width)] for row in range(height)]
        else:
            self.board = [[0 for x in range(width)] for i in range(height)]

        if (existing_board):
            self.first_empty_in_col = [first_empty for first_empty in existing_board.first_empty_in_col]
        else:
            self.first_empty_in_col = [0 for col in range(width)]

        self.winner = None

        if (existing_board):
            self.pieces = {}
            for player in [-1, 1]:
                self.pieces[player] = existing_board.get_piece_locations(player)
        else:
            self.pieces = {-1: [], 1: []}

    def get_column(self, index):
        return [i[index] for i in self.board]

    def get_row(self, index):
        return self.board[index]

    def is_on_board(self, row, col):
        return (row >= 0 and row < self.height) and (col >= 0 and col < self.width)

    def get_space(self, row, col):
        return self.board[row][col]

    # Return a deep copy of the current board
    def clone_board(self):
        return ConnectXBoard(height = self.height, width = self.width, x = self.x, existing_board = self)

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

    def check_for_win(self, row, col):
        # If the board is full, tie
        if (len(self.get_available_moves()) <= 0): return 0
        starts_and_offsets = [
                                ((row, 0), (0,1)), 
                                ((0, col), (1,0))
                                ]
        # weird stuff to make diagonals work
        if (row > col):
            starts_and_offsets.append((((row - col), 0), (1,1)))
            starts_and_offsets.append((((row + col), 0), (-1,1)))
        else:
            starts_and_offsets.append(((0, (col - row)), (1,1)))
            starts_and_offsets.append(((0, (col - row)), (-1,1)))

        for (start, offset) in starts_and_offsets:
            winner = self.check_row(start, offset)
            if (winner != None):
                return winner
        return None
    
    def check_row(self, start_pos, offset):
        tracked_piece = None
        num_in_a_row = 0
        curr_row, curr_col = start_pos
        row_offset, col_offset = offset
        while self.is_on_board(curr_row, curr_col):
            space = self.get_space(curr_row, curr_col)
            # If the new piece is the same as the last one then increase the number in a row
            if (tracked_piece == None or tracked_piece == space):
                num_in_a_row += 1
            else:
                # If the new one is different then the previous, reset
                num_in_a_row = 1
            # If there are enough in a row to win, then return the winner
            if (tracked_piece != None and tracked_piece != 0 and num_in_a_row >= self.x):
                return tracked_piece
            tracked_piece = space
            # Move to the next space
            curr_row += row_offset
            curr_col += col_offset
        return None

    # Gets the columns that can have a piece placed in them
    def get_available_moves(self):
        return [col for col in range(self.width) if self.col_available(col)]

    def get_empty_index_in_col(self, col):
        return self.first_empty_in_col[col]

    # Places a piece in the given row, doesn't mutate the board but instead returns a new board
    def make_move(self, col, player):
        if not self.col_available(col):
            raise RuntimeError("Column is full")
        # Find the first empty row in the column
        row_index = self.get_empty_index_in_col(col)
        new_board = self.clone_board()
        new_board.set_space(row_index, col, player)
        new_board.pieces[player].append((row_index, col))
        # Check if anyone won
        win = new_board.check_for_win(row_index, col)
        # If someone won set them as the winner
        if (win != None):
            new_board.winner = win
        return new_board


    def to_array(self):
        return self.board

    def to_vector(self):
        # Flatten the board into a vector (1D array)
        board_vector = [item for row in self.to_array() for item in row]
        return board_vector

    def to_tensor(self):
        # Flatten the board into a vector (1D array) and convert it to a pytorch tensor
        board_tensor = torch.tensor(self.to_vector()).float()
        return board_tensor

    def get_piece_locations(self, player):
        return self.pieces[player]

    def to_string(self):
        board_string = ""
        visual_representations = {1:'X', 0:'_', -1:'O'}
        for (line_num, line) in enumerate(self.to_array()):
            line = [visual_representations[item] for item in line]
            board_string = f"{line_num + 1}:  {'|'.join(line)}\n{board_string}"
        board_string += f"    {'|'.join([str(row_num) for row_num in range(1,self.width+1)])}"
        return board_string

if __name__ == '__main__':
    # start_game()
    board = ConnectXBoard()
    #print(game.board)
    new_board = board.make_move(4, 1)
    # new_board = new_board.make_move(6, 1)
    print(new_board.winner)
    new_board = new_board.make_move(1, -1)
    new_board = new_board.make_move(4, -1)
    new_board = new_board.make_move(4, -1)
    new_board = new_board.make_move(4, -1)
    # new_board = new_board.make_move(0, 1)
    print(new_board.winner)
    new_board = new_board.make_move(3, -1)
    new_board = new_board.make_move(3, -1)
    new_board = new_board.make_move(3, -1)
    # new_board = new_board.make_move(5, 1)
    print(new_board.winner)
    new_board = new_board.make_move(2, -1)
    new_board = new_board.make_move(2, -1)
    print(new_board.to_string())
    print(new_board.winner)
