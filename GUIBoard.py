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
GUIcursor = 0
running = True
pygame.init()


class GUIBoard(ConnectXBoard):
    def __init__(self, connectXBoard:ConnectXBoard = None) -> None:
        super().__init__()
        if connectXBoard is not None:  # there is existing board
            self.board = connectXBoard.board
            self.first_empty_in_col = connectXBoard.first_empty_in_col
            self.pieces = connectXBoard.pieces
        self.screenWidth = TILESIZE * self.width
        self.screenHeight = TILESIZE * (self.height + 2)
        self.dim = (self.screenHeight, self.screenWidth)
        self.screen = pygame.display.set_mode(self.dim)
        self.drawGUIboard(connectXBoard)


    def drawGUIboard(self, connectXBoard=None):
        # background
        self.screen.fill(BLACK)

        # arrow
        pygame.draw.polygon(self.screen, WHITE,
                            ((TILESIZE * (GUIcursor + 1) - 5, 50), (TILESIZE * (GUIcursor + 1) + 6.5, 50),
                             (TILESIZE * (GUIcursor + 1) + 5.5, 75), (TILESIZE * (GUIcursor + 1) + 20, 75),
                             (TILESIZE * (GUIcursor + 1) + 1.25, 93.75), (TILESIZE * (GUIcursor + 1) - 16.5, 75),
                             (TILESIZE * (GUIcursor + 1) - 5, 75)))
        # pygame.display.update()

        # board
        for row in range(self.height):
            for col in range(self.width):
                pygame.draw.rect(self.screen, BLUE,
                                 (TILESIZE * col + 50, TILESIZE + TILESIZE * row, TILESIZE, TILESIZE))
                pygame.draw.circle(self.screen, BLACK,
                                   ((TILESIZE * col + TILESIZE / 2) + 50, TILESIZE * row + TILESIZE / 2 + TILESIZE),
                                   DISKRAD)
        pygame.display.update()

        if connectXBoard != None:
            connectXBoard.board = np.flipud(connectXBoard.board)
            for row in range(self.height):
                for col in range(self.width):
                    # TODO draw disks
                    if connectXBoard.get_space(row, col) == -1:  # player
                        pygame.draw.circle(self.screen, RED,
                                           ((TILESIZE * col + TILESIZE / 2) + 50, TILESIZE * row + TILESIZE / 2 + TILESIZE),
                                           DISKRAD)
                        pygame.display.update()

                    elif connectXBoard.get_space(row, col) == 1:  # AI
                        pygame.draw.circle(self.screen, YELLOW,
                                           ((TILESIZE * col + TILESIZE / 2) + 50,
                                            TILESIZE * row + TILESIZE / 2 + TILESIZE),
                                           DISKRAD)
                        pygame.display.update()

        if connectXBoard != None:
            connectXBoard.board = np.flipud(connectXBoard.board)
        pygame.display.update()


if __name__ == '__main__':
    test_board = ConnectXBoard(height=6, width=7, x=4)
    humanPlayer = HumanConnect4Player()
    gui = GUIBoard()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:  # when key is pressed
                if event.key == pygame.K_LEFT:
                    if 0 < GUIcursor < gui.width:
                        GUIcursor -= 1

                if event.key == pygame.K_RIGHT:
                    if 0 <= GUIcursor < gui.width - 1:
                        GUIcursor += 1

                if event.key == pygame.K_SPACE:
                    # TODO: space key. make move if valid
                    gui = GUIBoard(gui.make_move(GUIcursor, humanPlayer.player))
                    print(gui.to_string())
                    # print(gui.to_array())

            # pygame.display.update()
