from RandomConnect4Player import RandomConnect4Player
from GUIBoard import GUIBoard
from ConnectXBoard import ConnectXBoard
from AlphaBetaConnect4Player import AlphaBetaConnect4Player
from ConnectXHeuristics import partial_lines_heuristic, dist_heuristic, table_heuristic
from DQNConnect4Player import DQNConnect4Player
from ConnectXDQNTrainer import DQNAgent

from ModelLoading import load_model

import pygame
import sys
import time

class Connect4Game:

    def __init__(self, board, player_two, move_delay = 0):
        self.player_two = player_two
        self.base_board = board.clone_board()
        self.gui = GUIBoard(self.base_board.clone_board())
        self.move_delay = move_delay
        self.game()

    def game(self):
        running = True
        while running:
            self.gui.draw_board()
            if (self.gui.board.winner == None):
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
                            if (self.gui.cursor_loc in self.gui.board.get_available_moves()):
                                self.gui.board = self.gui.board.make_move(self.gui.cursor_loc, -1)
                                if (self.gui.board.winner == None):
                                    p2_move = self.player_two.get_move(self.gui.board, 1)
                                    self.gui.board = self.gui.board.make_move(p2_move, 1)
                        # print(gui.to_array())
            else:
                time.sleep(3)
                running = False
                return

if __name__ == '__main__':
    test_board = ConnectXBoard()

    other_player = DQNConnect4Player(load_model('10000-eps-respective_powered'))

    while True:
        print('New Game')
        game = Connect4Game(test_board, other_player, move_delay=0.5)
