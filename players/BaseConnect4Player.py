# The base class that all of the Connect4Players will use
class BaseConnect4Player():
    def __init__(self):
        pass

    def get_move(self, board):
        # Picks the best move for the given board, return the column number for the move
        raise NotImplementedError("This function must be overidden by child classes")