# This class will use a DQN to make moves for Connect 4
class DQNConnect4Player():
    def __init__(self, dqn_agent):
        self.dqn_agent = dqn_agent

    def get_move(self, board, player):
        # Picks the best move for the given board, return the column number for the move
        best_move = self.dqn_agent.get_move_private(board.to_tensor() * player)
        while best_move not in board.get_available_moves():
            print('move')
            print(best_move)
            print('valid moves')
            print(board.get_available_moves())
            best_move = self.dqn_agent.get_move_private(board.to_tensor() * player)
        return best_move