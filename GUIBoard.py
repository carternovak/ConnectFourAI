import numpy as np
import pygame
import sys
from ConnectXBoard import *
from HumanConnect4Player import HumanConnect4Player

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (36, 84, 196)
YELLOW = (239, 226, 40)
RED = (222, 37, 0)
TILESIZE = 100
DISKRAD = int(TILESIZE / 2 - 5)
#GUIcursor = 0
running = True
pygame.init()


class GUIBoard(ConnectXBoard):
    def __init__(self, connectXBoard: ConnectXBoard = None) -> None:
        super().__init__()
        if connectXBoard is not None:  # there is existing board
            self.board = connectXBoard
            self.first_empty_in_col = connectXBoard.first_empty_in_col
            #self.pieces = connectXBoard.pieces
        self.screenWidth = TILESIZE * self.width
        self.screenHeight = TILESIZE * (self.height + 2)
        self.dim = (self.screenHeight, self.screenWidth)
        self.screen = pygame.display.set_mode(self.dim)
        self.cursor_loc = 0

    def draw_board(self):
        self.draw_background()
        self.draw_pieces()
        self.draw_arrow()
        pygame.display.update()

    def draw_pieces(self):
        # self.board = np.flipud(self.board.to_array())
        for row in range(self.board.height):
            for col in range(self.board.width):
                piece = self.board.get_space(row, col)
                # Unflip the rows
                row = (self.board.height - row - 1)
                if piece == -1:  # player
                    pygame.draw.circle(self.screen, RED,
                                       ((TILESIZE * col + TILESIZE / 2) + 50,
                                        TILESIZE * row + TILESIZE / 2 + TILESIZE),
                                       DISKRAD)
                elif piece == 1:  # AI
                    pygame.draw.circle(self.screen, YELLOW,
                                       ((TILESIZE * col + TILESIZE / 2) + 50,
                                        TILESIZE * row + TILESIZE / 2 + TILESIZE),
                                       DISKRAD)

    def draw_arrow(self):
        pygame.draw.polygon(self.screen, WHITE,
                            ((TILESIZE * (self.cursor_loc + 1) - 5, 50), (TILESIZE * (self.cursor_loc + 1) + 6.5, 50),
                             (TILESIZE * (self.cursor_loc + 1) + 5.5, 75), (TILESIZE * (self.cursor_loc + 1) + 20, 75),
                             (TILESIZE * (self.cursor_loc + 1) + 1.25, 93.75), (TILESIZE * (self.cursor_loc + 1) - 16.5, 75),
                             (TILESIZE * (self.cursor_loc + 1) - 5, 75)))

    def draw_background(self):
        self.screen.fill(BLACK)
        for row in range(self.board.height):
            for col in range(self.board.width):
                pygame.draw.rect(self.screen, BLUE,
                                 (TILESIZE * col + 50, TILESIZE + TILESIZE * row, TILESIZE, TILESIZE))
                pygame.draw.circle(self.screen, BLACK,
                                   ((TILESIZE * col + TILESIZE / 2) + 50, TILESIZE * row + TILESIZE / 2 + TILESIZE),
                                   DISKRAD)


if __name__ == '__main__':
    test_board = ConnectXBoard(height=6, width=7, x=4)
    humanPlayer = HumanConnect4Player()
    gui = GUIBoard(test_board)

    while running:
        gui.draw_board()
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:  # when key is pressed
                if event.key == pygame.K_LEFT:
                    if 0 < gui.cursor_loc < gui.width:
                        gui.cursor_loc -= 1

                if event.key == pygame.K_RIGHT:
                    if 0 <= gui.cursor_loc < gui.width - 1:
                        gui.cursor_loc += 1

                if event.key == pygame.K_SPACE:
                    gui.board = gui.board.make_move(gui.cursor_loc, -1)
                    gui.draw_pieces()
                    # print(gui.to_string())
