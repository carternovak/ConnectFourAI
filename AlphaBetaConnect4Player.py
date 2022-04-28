# Connect4Player that uses MinMax with Alpha Beta to find the best moves
from ast import Raise
import time
import math

class AlphaBetaConnect4Player():
    def __init__(self, heuristic, player = 1, max_depth = None, max_time = None):
        self.heuristic = heuristic
        self.max_depth = max_depth
        self.max_time = max_time
        self.player = player

    def get_move(self, board, player):
        # Picks the best move for the given board, return the column number for the move
        best_move = self.get_best_move_alpha_beta(board, player)
        return best_move

    def get_best_move_alpha_beta(self, board, player):
        start_time = time.time()
        start_alpha = math.inf
        start_beta = -math.inf
        return self.negamax_ab(board, 0, start_alpha, start_beta, player, start_time)[1]

    # Uses negamax, a variant of minmax (https://en.wikipedia.org/wiki/Negamax) along with alpha beta
    def negamax_ab(self, board, alpha, beta, depth, player, start_time):
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
            board_after_move = board.make_move(col, self.player)
            move_value, _ = self.negamax_ab(board_after_move,  depth + 1, -beta, -alpha, player, start_time)
            move_value = -move_value
            if (move_value > value):
                best_move = col
                value = move_value

            alpha = max(alpha, move_value)
            if alpha >= beta:
                # Stop searching
                break

        return value, best_move