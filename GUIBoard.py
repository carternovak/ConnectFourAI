import pygame
from ConnectXBoard import *

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (36, 84, 196)
YELLOW = (239, 226, 40)
RED = (222, 37, 0)
TILESIZE = 100
class GUIBoard:

    def __init__(self, board) -> None:
        # GUI
        self.board = board
        self.running = True
        pygame.init()
        self.diskRad = int(TILESIZE/2 - 5)
        self.screenWidth = TILESIZE * board.width
        self.screenHeight = TILESIZE * (board.height + 2)
        self.dim = (self.screenHeight, self.screenWidth)
        self.screen = pygame.display.set_mode(self.dim)
        self.drawGUIboard()
        # pygame.time.wait(5000)

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