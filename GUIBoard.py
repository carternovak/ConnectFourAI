import pygame
import sys
from ConnectXBoard import *

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (36, 84, 196)
YELLOW = (239, 226, 40)
RED = (222, 37, 0)
TILESIZE = 100
DISKRAD = int(TILESIZE / 2 - 5)
CURSOR = 0
RUNNING = True
pygame.init()

class GUIBoard:
    def __init__(self, board) -> None:
        # GUI
        self.board = board
        self.diskRad = int(TILESIZE/2 - 5)
        self.screenWidth = TILESIZE * board.width
        self.screenHeight = TILESIZE * (board.height + 2)
        self.dim = (self.screenHeight, self.screenWidth)
        self.screen = pygame.display.set_mode(self.dim)
        self.drawGUIboard()

    def drawGUIboard(self):
        for col in range(self.board.width):
            for row in range(self.board.height):
                pygame.draw.rect(self.screen, BLUE, (TILESIZE * col, TILESIZE + TILESIZE*row, TILESIZE, TILESIZE))
                pygame.draw.circle(self.screen, WHITE, ((TILESIZE * col + TILESIZE/2), TILESIZE*row  + TILESIZE/2 + TILESIZE), self.diskRad)
        pygame.display.update()

if __name__ == '__main__':
    test_board = ConnectXBoard(height=7, width=6, x=4)
    gui = GUIBoard(test_board)
    gui.drawGUIboard()
    while RUNNING:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:    # when key is pressed
                if event.key == pygame.K_LEFT:
                    # TODO right key. move to left if valid
                    print("left key!")

                if event.type == pygame.K_RIGHT:
                    # TODO: right key. move to right if valid
                    print("right key!")

                if event.key == pygame.K_SPACE:
                    # TODO: space key. make move if valid
                    print("space key!")
            gui.drawGUIboard()
            # pygame.display.update()