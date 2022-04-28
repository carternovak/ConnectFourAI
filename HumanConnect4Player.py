from BaseConnect4Player import *
# The human interface for playing Connect 4

class HumanConnect4Player(BaseConnect4Player):
    def __init__(self):
        super().__init__()
        self.player = -1
        pass

    def get_move(self, board, player):
        # Get user input and return it
        print('Board:')
        print(board.to_string())
        try:
            col = int(input("Which column would you like to place a piece in?"))
        except:
            col = -1
        while col - 1 not in board.get_available_moves():
            try:
                col = int(input(f"Column {col} is full, Which column would you like to place a piece in?"))
            except:
                col = -1
        return col - 1
