import BaseConnect4Player
# The human interface for playing Connect 4
class HumanConnect4Player(BaseConnect4Player):
    def __init__(self):
        pass

    def get_move(self, board):
        # Get user input and return it
        col = input("Which column would you like to place a piece in?")
        while not board.col_available(col - 1):
            col = input(f"Column {col} is full, Which column would you like to place a piece in?")
        return col - 1
