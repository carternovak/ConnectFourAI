import random
# The base class that all of the Connect4Players will use
class RandomConnect4Player():
    def __init__(self):
        pass

    def get_move(self, board, player):
        # Returns a random valid move
        valid_moves = board.get_available_moves()
        return valid_moves[random.randrange(0, len(valid_moves))]