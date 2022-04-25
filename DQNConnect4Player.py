# This class will use a DQN to make moves for Connect 4
class DQNConnect4Player():
    def __init__(self, dqn_agent, conv=False):
        self.dqn_agent = dqn_agent
        self.conv = conv

    def get_move(self, board):
        # Picks the best move for the given board, return the column number for the move
        board_tensor = board.to_tensor()
        best_move = self.dqn_agent.get_move(board_tensor)
        while best_move not in board.get_available_moves():
            best_move = self.dqn_agent.get_move(board.to_tensor())
        return best_move