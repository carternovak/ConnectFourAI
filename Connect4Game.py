from GUIBoard import GUIBoard
from ConnectXBoard import ConnectXBoard
from AlphaBetaConnect4Player import AlphaBetaConnect4Player
from ConnectXHeuristics import partial_lines_heuristic, dist_heuristic
from DQNConnect4Player import DQNConnect4Player
from ConnectXDQNTrainer import DQNAgent
import torch
import pygame
import sys
import time

class Connect4Game:

    def __init__(self, board, player_two, move_delay = 0):
        # self.human = player_one == 'human'
        # self.player_one = player_one
        self.player_two = player_two
        self.base_board = board.clone_board()
        self.gui = GUIBoard(self.base_board.clone_board())
        self.move_delay = move_delay
        self.game()

    def game(self):
        running = True
        while running:
            self.gui.drawGUIboard()
            if (self.gui.board.winner == None):
                self.gui.drawGUIboard()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                    if event.type == pygame.KEYDOWN:  # when key is pressed
                        if event.key == pygame.K_LEFT:
                            if 0 < self.gui.cursor_loc < self.gui.board.width:
                                self.gui.cursor_loc -= 1

                        if event.key == pygame.K_RIGHT:
                            if 0 <= self.gui.cursor_loc < self.gui.board.width - 1:
                                self.gui.cursor_loc += 1

                        if event.key == pygame.K_SPACE:
                            # TODO: space key. make move if valid
                            print(self.gui.cursor_loc)
                            if (self.gui.cursor_loc in self.gui.board.get_available_moves()):
                                self.gui.board = self.gui.board.make_move(self.gui.cursor_loc, -1)
                                if (self.gui.board.winner == None):
                                    p2_move = self.player_two.get_move(self.gui.board)
                                    self.gui.board = self.gui.board.make_move(p2_move, 1)
                        # print(gui.to_array())
            else:
                time.sleep(3)
                running = False
                return

if __name__ == '__main__':
    test_board = ConnectXBoard()
    good_conv = DQNAgent(None, 
                        conv_model=True, 
                        board_width=test_board.width, 
                        board_height=test_board.height, 
                        action_states=test_board.width, 
                        batch_size=100, 
                        epsilon=.999, 
                        epsilon_decay=0.01, 
                        min_epsilon=0.05, 
                        gamma=.5, 
                        lr=0.0001,
                        pre_trained_policy=torch.load('good-dqn-policy.pt'),
                        pre_trained_target=torch.load('good-dqn-target.pt'))
    while True:
        print('New Game')
        game = Connect4Game(test_board, DQNConnect4Player(good_conv), move_delay=0.5)
