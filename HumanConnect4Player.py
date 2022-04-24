from BaseConnect4Player import *
# The human interface for playing Connect 4

class HumanConnect4Player(BaseConnect4Player):
    def __init__(self):
        pass

    def get_move(self, board):
        # Get user input and return it
        print('Board:')
        print(board.to_string())
        col = int(input("Which column would you like to place a piece in?"))
        while col - 1 not in board.get_available_moves():
            col = int(input(f"Column {col} is full, Which column would you like to place a piece in?"))
        return col - 1
