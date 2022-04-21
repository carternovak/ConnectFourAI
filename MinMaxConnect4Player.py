# Connect4Player that uses MinMax to find the best moves
import time
import math
from unittest.mock import NonCallableMagicMock

class MinMaxConnect4Player():
    def __init__(self, heuristic, max_depth = None, max_time = None):
        self.heuristic = heuristic
        self.max_depth = max_depth
        self.max_time = max_time

    def get_move(self, board):
        # Picks the best move for the given board, return the column number for the move
        raise NotImplementedError("This function must be overidden by child classes")

    def get_best_move(self):
        start_time = time.time()
        pass

    # Uses negamax, a variant of minmax (https://en.wikipedia.org/wiki/Negamax)
    def negamax(self, board, depth, player, start_time):
        if (depth > self.max_depth) or (time.time() - start_time > self.max_time):
            return player * self.heuristic(board)
        if (board.winner != 0):
            return player * board.winner
        value = -math.inf
        for col in board.get_available_moves():
            board_after_move = board.make_move(col)
            value = max(value, -self.negamax(board_after_move, depth + 1, player, start_time))

