import gym
from gym import spaces

class ConnectXGym(gym.Env):

    def __init__(self, board, other_player, player_num):
        # Stores the base values of the board (used later to reset the environment)
        self.base_board = board
        self.board = self.base_board.clone_board()
        # The action spaces is each column you can put soemthing in
        self.action_space = spaces.Discrete(self.board.width)
        # The obeservation space is the state of the board respresented as 1D vector
        # Sot the length of the vector is width * height
        obs_space_size = self.board.width * self.board.height
        low = [-1 for _ in range(obs_space_size)]
        high = [1 for _ in range(obs_space_size)]
        self.observation_space = spaces.Box(low=low, high=high)
        self.other_player = other_player
        self.player_num = player_num

    def step(self, action):
        # The action represents the column to put a piece into
    
        if (not self.board.is_row_open(action)):
            # If an invalid move was made, very bad
            reward = -10
            info = {"Reason":"Invalid mode attempted"}
        else:
            # Make the move
            self.board = self.board.make_move(action)
            if (self.board.winner == self.player_num):
                # If you won very good
                reward = 100
                info = {"Reason":"Won"}
            else:
                # If not give a small bonus for staying alive
                reward = 1
                info = {"Reason":"Didn't Lose"}
            other_player_move = self.other_player.get_move()
            self.board = self.board.make_move(other_player_move)
        
        state = self.board.to_vector()
        done = self.board.winner != None
        return state, reward, done, info

    def render(self, mode = "human"):
        if (mode == "human"):
            print(self.board.to_array())

    def reset(self):
            self.board = self.base_board.clone_board()
            return self.board.to_vector()