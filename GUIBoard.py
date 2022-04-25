from cmath import pi
import pygame
import sys
from ConnectXBoard import *
from HumanConnect4Player import *

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
    def __init__(self, board) -> None:
        super().__init__()
        self.screenWidth = TILESIZE * (board.width + 1)
        self.screenHeight = (TILESIZE * (board.height + 2))
        self.dim = (self.screenWidth, self.screenHeight)
        self.screen = pygame.display.set_mode(self.dim)
        self.board = board

    def drawGUIboard(self):
        # background
        self.screen.fill(BLACK)

        # arrow
        pygame.draw.polygon(self.screen, WHITE,
                            ((TILESIZE * (GUIcursor+1)-5, 50),             (TILESIZE * (GUIcursor+1) + 6.5, 50),
                             (TILESIZE * (GUIcursor+1) + 5.5, 75),      (TILESIZE * (GUIcursor+1) + 20, 75),
                             (TILESIZE * (GUIcursor+1) + 1.25, 93.75),   (TILESIZE * (GUIcursor+1) - 16.5, 75),
                             (TILESIZE * (GUIcursor+1)-5, 75)))

        # board
        for row in range(self.board.height):
            for col in range(self.board.width):
                piece = self.board.get_space(row, col)
                row = (self.board.height - 1 - row)
                colors = {0:BLACK, 1:YELLOW, -1:RED}
                color = colors[piece]
                pygame.draw.rect(self.screen, BLUE,
                                 (TILESIZE * col + 50, TILESIZE + TILESIZE * row, TILESIZE, TILESIZE))
                pygame.draw.circle(self.screen, color,
                                   ((TILESIZE * col + TILESIZE / 2) + 50, TILESIZE * row + TILESIZE / 2 + TILESIZE),
                                   DISKRAD)
        # for row in range(self.height):
        #     for col in range(self.width):
        #         TODO draw disks

        pygame.display.update()


if __name__ == '__main__':
    test_board = ConnectXBoard(height=7, width=6, x=4)
    player = 1
    gui = GUIBoard(test_board)

    while running:
        gui.drawGUIboard()
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
                    gui.make_move(GUIcursor, player)
                    print(gui.to_string())
                    print(gui.to_array())

            # pygame.display.update()
