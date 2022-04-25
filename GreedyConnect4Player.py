# The always places a piece in the first available spot
class GreedyConnect4Player():
    def __init__(self):
        pass

    def get_move(self, board):
        # Returns a random valid move
        valid_moves = board.get_available_moves()
        return valid_moves[0]