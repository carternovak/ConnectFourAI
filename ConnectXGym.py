import random
import gym
from gym import spaces
from GUIBoard import *

class ConnectXGym(gym.Env):

    def __init__(self, board, other_player, player_num, other_player_num):
        # Stores the base values of the board (used later to reset the environment)
        self.base_board = board
        self.board = self.base_board.clone_board()
        # The action spaces is each column you can put something in
        self.action_space = spaces.Discrete(self.board.width)
        # The obeservation space is the state of the board respresented as 1D vector
        # Set the length of the vector is width * height
        obs_space_size = self.board.width * self.board.height
        self.observation_space = spaces.MultiDiscrete([3 for _ in range(obs_space_size)])
        self.other_player = other_player
        self.player_num = player_num
        self.other_player_num = other_player_num

    def step(self, action):
        # The action represents the column to put a piece into
    
        reward = 0
        info = {"Reason":"Nothing Special"}

        if (not self.board.col_available(action)):
            # If an invalid move was made, very bad
            print("Illegal Move Made")
            # raise RuntimeError("Illegal Move Made")
        else:
            # Make the move
            self.board = self.board.make_move(action, self.player_num)

            # Only make the opponent move if you haven't already won
            if (self.board.winner == None):
                other_player_move = self.other_player.get_move(self.board, self.other_player_num)
                self.board = self.board.make_move(other_player_move, self.other_player_num)

        if (self.board.winner == None):
            reward = 0
            info = {"Reason":"Nothing special"}
        else:
            if (self.board.winner == self.player_num):
                # If you won very good
                reward = 1
                info = {"Reason":"Won"}
            if (self.board.winner == 0):
                # If you forced tie, ok
                reward = 0
                info = {"Reason":"Move resulted in tie"}
            if (self.board.winner == self.other_player_num):
                # If you allowed your opponent to win, bad
                reward = -1
                info = {"Reason":"Move resulted in other player winning"}
        
        # Return the board as a tensor along with the reward, whether the game is over, and any other info
        state = self.board.to_tensor()
        done = self.board.winner != None
        return state, reward, done, info

    def render(self, mode = "human"):
        if (mode == "human"):
            # Render the board
            GUIBoard(self.board).drawGUIboard()
        if (mode == "tensor"):
            return self.board.to_tensor()

    def reset(self):
            self.board = self.base_board.clone_board()
            player_going_first = random.randint(1,2)
            # If the other player is going first then make a move for them
            if (player_going_first == 2):
                other_player_move = self.other_player.get_move(self.board, self.other_player_num)
                self.board = self.board.make_move(other_player_move, self.other_player_num)
                
            return self.board.to_tensor()