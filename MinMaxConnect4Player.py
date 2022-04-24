# Connect4Player that uses MinMax to find the best moves
import time
import math

class MinMaxConnect4Player():
    def __init__(self, heuristic, player = 1, max_depth = None, max_time = None):
        self.heuristic = heuristic
        self.max_depth = max_depth
        self.max_time = max_time
        self.player = player

    def get_move(self, board):
        # Picks the best move for the given board, return the column number for the move
        return self.get_best_move_min_max(board)

    def get_best_move_min_max(self, board):
        start_time = time.time()
        return self.negamax(board, 0, self.player, start_time)[1]

    # Uses negamax, a variant of minmax (https://en.wikipedia.org/wiki/Negamax)
    def negamax(self, board, depth, player, start_time):
        # If the horizon has been reached then use the heuristic
        best_move = None
        too_far_down = self.max_depth and (depth > self.max_depth)
        too_long = self.max_time and (time.time() - start_time > self.max_time)
        if too_far_down or too_long:
            return player * self.heuristic(board), best_move
        if (board.winner != None):
            return player * board.winner, best_move
        value = -math.inf
        for col in board.get_available_moves():
            board_after_move = board.make_move(col, player)
            move_value, _ = self.negamax(board_after_move, depth + 1, player, start_time)
            move_value = -move_value
            if (move_value > value):
                best_move = col
                value = move_value
        return value, best_move