# This class will use a DQN to make moves for Connect 4
class DQNConnect4Player():
    def __init__(self, dqn_predictor):
        self.dqn_predictor = dqn_predictor

    def get_move(self, board):
        # Picks the best move for the given board, return the column number for the move
        best_move = self.dqn_predictor(board)
        raise NotImplementedError("DQN is not implemented yet")