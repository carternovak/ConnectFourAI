import random
import numpy as np
#from collections import Set

def respective_powered(self, board):
    finalEval = 0
    finalEval += (fourinarow(board)*4)**4
    finalEval += (threeinarow(board)*3)**3
    finalEval += (twoinarow(board)*2)**2
    return finalEval

def fourinarow(board):
    p1_four = 0
    p2_four = 0
    #horizontal
    for i in board.height:
        for j in board.width - 3:
            if board[i][j] == 1 and board[i][j+1] == 1 and board[i][j+2] == 1 and board[i][j+3] == 1:
                p1_four += 1
            elif board[i][j] == -1 and board[i][j+1] == -1 and board[i][j+2] == -1 and board[i][j+3] == -1:
                p2_four -= 1
    #vertical
    for i in board.height - 3:
        for j in board.width:
            if board[i][j] == 1 and board[i+1][j] == 1 and board[i+2][j] == 1 and board[i+3][j] == 1:
                p1_four += 1
            elif board[i][j] == -1 and board[i+1][j] == -1 and board[i+2][j] == -1 and board[i+3][j] == -1:
                p2_four -= 1
    #negative diagonal
    for i in board.height:
        for j in board.width - 3:
            if board[i][j] == 1 and board[i-1][j+1] == 1 and board[i-2][j+2] == 1 and board[i-3][j+3] == 1:
                p1_four += 1
            elif board[i][j] == -1 and board[i-1][j+1] == -1 and board[i-2][j+2] == -1 and board[i-3][j+3] == -1:
                p2_four -= 1
    #positive diagonal
    for i in board.height - 3:
        for j in board.width - 3:
            if board[i][j] == 1 and board[i+1][j+1] == 1 and board[i+2][j+2] == 1 and board[i+3][j+3] == 1:
                p1_four += 1
            elif board[i][j] == -1 and board[i+1][j+1] == -1 and board[i+2][j+2] == -1 and board[i+3][j+3] == -1:
                p2_four -= 1

    return max(p1_four, abs(p2_four))

def threeinarow(board):
    p1_three = 0
    p2_three = 0
    #horizontal
    for i in board.height:
        for j in board.width - 3:
            if board[i][j] == 1 and board[i][j+1] == 1 and board[i][j+2] == 1 and board[i][j+3] == 0:
                p1_three += 1
            elif board[i][j] == -1 and board[i][j+1] == -1 and board[i][j+2] == -1 and board[i][j+3] == 0:
                p2_three += 1
            if board[i][j] == 1 and board[i][j+1] == 1 and board[i][j+2] == 0 and board[i][j+3] == 1:
                p1_three += 1
            elif board[i][j] == -1 and board[i][j+1] == -1 and board[i][j+2] == 0 and board[i][j+3] == -1:
                p2_three += 1
            if board[i][j] == 1 and board[i][j+1] == 0 and board[i][j+2] == 1 and board[i][j+3] == 1:
                p1_three += 1
            elif board[i][j] == -1 and board[i][j+1] == 0 and board[i][j+2] == -1 and board[i][j+3] == -1:
                p2_three += 1
            if board[i][j] == 0 and board[i][j+1] == 1 and board[i][j+2] == 1 and board[i][j+3] == 1:
                p1_three += 1
            elif board[i][j] == 0 and board[i][j+1] == -1 and board[i][j+2] == -1 and board[i][j+3] == -1:
                p2_three += 1
    #vertical
    for i in board.height - 3:
        for j in board.width:
            if board[i][j] == 1 and board[i+1][j] == 1 and board[i+2][j] == 1 and board[i+3][j] == 0:
                p1_three += 1
            elif board[i][j] == -1 and board[i+1][j] == -1 and board[i+2][j] == -1 and board[i+3][j] == 0:
                p2_three += 1
            if board[i][j] == 1 and board[i+1][j] == 1 and board[i+2][j] == 0 and board[i+3][j] == 1:
                p1_three += 1
            elif board[i][j] == -1 and board[i+1][j] == -1 and board[i+2][j] == 0 and board[i+3][j] == -1:
                p2_three += 1
            if board[i][j] == 1 and board[i+1][j] == 0 and board[i+2][j] == 1 and board[i+3][j] == 1:
                p1_three += 1
            elif board[i][j] == -1 and board[i+1][j] == 0 and board[i+2][j] == -1 and board[i+3][j] == -1:
                p2_three += 1
            if board[i][j] == 0 and board[i+1][j] == 1 and board[i+2][j] == 1 and board[i+3][j] == 1:
                p1_three += 1
            elif board[i][j] == 0 and board[i+1][j] == -1 and board[i+2][j] == -1 and board[i+3][j] == -1:
                p2_three += 1
    # negative diagonal
    for i in board.height:
        for j in board.width-3:
            if board[i][j] == 1 and board[i-1][j + 1] == 1 and board[i-2][j + 2] == 1 and board[i-3][j + 3] == 0:
                p1_three += 1
            elif board[i][j] == -1 and board[i-1][j + 1] == -1 and board[i-2][j + 2] == -1 and board[i-3][j + 3] == 0:
                p2_three += 1
            if board[i][j] == 1 and board[i-1][j + 1] == 1 and board[i-2][j + 2] == 0 and board[i-3][j + 3] == 1:
                p1_three += 1
            elif board[i][j] == -1 and board[i-1][j + 1] == -1 and board[i-2][j + 2] == 0 and board[i-3][j + 3] == -1:
                p2_three += 1
            if board[i][j] == 1 and board[i-1][j + 1] == 0 and board[i-2][j + 2] == 1 and board[i-3][j + 3] == 1:
                p1_three += 1
            elif board[i][j] == -1 and board[i-1][j + 1] == 0 and board[i-2][j + 2] == -1 and board[i-3][j + 3] == -1:
                p2_three += 1
            if board[i][j] == 0 and board[i-1][j + 1] == 1 and board[i-2][j + 2] == 1 and board[i-3][j + 3] == 1:
                p1_three += 1
            elif board[i][j] == 0 and board[i-1][j + 1] == -1 and board[i-2][j + 2] == -1 and board[i-3][j + 3] == -1:
                p2_three += 1
    # positive diagonal
    for i in board.height - 3:
        for j in board.width - 3:
            if board[i][j] == 1 and board[i + 1][j + 1] == 1 and board[i + 2][j + 2] == 1 and board[i + 3][j + 3] == 0:
                p1_three += 1
            elif board[i][j] == -1 and board[i + 1][j + 1] == -1 and board[i + 2][j + 2] == -1 and board[i + 3][j + 3] == 0:
                p2_three += 1
            if board[i][j] == 1 and board[i + 1][j + 1] == 1 and board[i + 2][j + 2] == 0 and board[i + 3][j + 3] == 1:
                p1_three += 1
            elif board[i][j] == -1 and board[i + 1][j + 1] == -1 and board[i + 2][j + 2] == 0 and board[i + 3][j + 3] == -1:
                p2_three += 1
            if board[i][j] == 1 and board[i + 1][j + 1] == 0 and board[i + 2][j + 2] == 1 and board[i + 3][j + 3] == 1:
                p1_three += 1
            elif board[i][j] == -1 and board[i + 1][j + 1] == 0 and board[i + 2][j + 2] == -1 and board[i + 3][j + 3] == -1:
                p2_three += 1
            if board[i][j] == 0 and board[i + 1][j + 1] == 1 and board[i + 2][j + 2] == 1 and board[i + 3][j + 3] == 1:
                p1_three += 1
            elif board[i][j] == 0 and board[i + 1][j + 1] == -1 and board[i + 2][j + 2] == -1 and board[i + 3][j + 3] == -1:
                p2_three += 1

def twoinarow(board):
    p1_two = 0
    p2_two = 0
    #horizontal
    for i in board.height:
        for j in board.width - 3:
            if board[i][j] == 1 and board[i][j+1] == 1 and board[i][j+2] == 0 and board[i][j+3] == 0:
                p1_two += 1
            elif board[i][j] == -1 and board[i][j+1] == -1 and board[i][j+2] == 0 and board[i][j+3] == 0:
                p2_two -= 1
            if board[i][j] == 1 and board[i][j+1] == 0 and board[i][j+2] == 1 and board[i][j+3] == 0:
                p1_two += 1
            elif board[i][j] == -1 and board[i][j+1] == 0 and board[i][j+2] == -1 and board[i][j+3] == 0:
                p2_two -= 1
            if board[i][j] == 0 and board[i][j+1] == 1 and board[i][j+2] == 1 and board[i][j+3] == 0:
                p1_two += 1
            elif board[i][j] == 0 and board[i][j+1] == -1 and board[i][j+2] == -1 and board[i][j+3] == 0:
                p2_two -= 1
            if board[i][j] == 0 and board[i][j+1] == 1 and board[i][j+2] == 0 and board[i][j+3] == 1:
                p1_two += 1
            elif board[i][j] == 0 and board[i][j+1] == -1 and board[i][j+2] == 0 and board[i][j+3] == -1:
                p2_two -= 1
            if board[i][j] == 0 and board[i][j+1] == 0 and board[i][j+2] == 1 and board[i][j+3] == 1:
                p1_two += 1
            elif board[i][j] == 0 and board[i][j+1] == 0 and board[i][j+2] == -1 and board[i][j+3] == -1:
                p2_two -= 1
    #vertical
    for i in board.height - 3:
        for j in board.width:
            if board[i][j] == 1 and board[i+1][j] == 1 and board[i+2][j] == 0 and board[i+3][j] == 0:
                p1_two += 1
            elif board[i][j] == -1 and board[i+1][j] == -1 and board[i+2][j] == 0 and board[i+3][j] == 0:
                p2_two -= 1
            if board[i][j] == 1 and board[i+1][j] == 0 and board[i+2][j] == 1 and board[i+3][j] == 0:
                p1_two += 1
            elif board[i][j] == -1 and board[i+1][j] == 0 and board[i+2][j] == -1 and board[i+3][j] == 0:
                p2_two -= 1
            if board[i][j] == 0 and board[i+1][j] == 1 and board[i+2][j] == 1 and board[i+3][j] == 0:
                p1_two += 1
            elif board[i][j] == 0 and board[i+1][j] == -1 and board[i+2][j] == -1 and board[i+3][j] == 0:
                p2_two -= 1
            if board[i][j] == 0 and board[i+1][j] == 1 and board[i+2][j] == 0 and board[i+3][j] == 1:
                p1_two += 1
            elif board[i][j] == 0 and board[i+1][j] == -1 and board[i+2][j] == 0 and board[i+3][j] == -1:
                p2_two -= 1
            if board[i][j] == 0 and board[i+1][j] == 0 and board[i+2][j] == 1 and board[i+3][j] == 1:
                p1_two += 1
            elif board[i][j] == 0 and board[i+1][j] == 0 and board[i+2][j] == -1 and board[i+3][j] == -1:
                p2_two -= 1
    #negative diagonal
    for i in board.height:
        for j in board.width - 3:
            if board[i][j] == 1 and board[i - 1][j + 1] == 1 and board[i - 2][j + 2] == 0 and board[i - 3][j + 3] == 0:
                p1_two += 1
            elif board[i][j] == -1 and board[i - 1][j + 1] == -1 and board[i - 2][j + 2] == 0 and board[i - 3][j + 3] == 0:
                p2_two -= 1
            if board[i][j] == 1 and board[i - 1][j + 1] == 0 and board[i - 2][j + 2] == 1 and board[i - 3][j + 3] == 0:
                p1_two += 1
            elif board[i][j] == -1 and board[i - 1][j + 1] == 0 and board[i - 2][j + 2] == -1 and board[i - 3][j + 3] == 0:
                p2_two -= 1
            if board[i][j] == 0 and board[i - 1][j + 1] == 1 and board[i - 2][j + 2] == 1 and board[i - 3][j + 3] == 0:
                p1_two += 1
            elif board[i][j] == 0 and board[i - 1][j + 1] == -1 and board[i - 2][j + 2] == -1 and board[i - 3][j + 3] == 0:
                p2_two -= 1
            if board[i][j] == 0 and board[i - 1][j + 1] == 1 and board[i - 2][j + 2] == 0 and board[i - 3][j + 3] == 1:
                p1_two += 1
            elif board[i][j] == 0 and board[i - 1][j + 1] == -1 and board[i - 2][j + 2] == 0 and board[i - 3][j + 3] == -1:
                p2_two -= 1
            if board[i][j] == 0 and board[i - 1][j + 1] == 0 and board[i - 2][j + 2] == 1 and board[i - 3][j + 3] == 1:
                p1_two += 1
            elif board[i][j] == 0 and board[i - 1][j + 1] == 0 and board[i - 2][j + 2] == -1 and board[i - 3][j + 3] == -1:
                p2_two -= 1
    #positive diagonal
    for i in board.height - 3:
        for j in board.width - 3:
            if board[i][j] == 1 and board[i+1][j+1] == 1 and board[i+2][j+2] == 0 and board[i+3][j+3] == 0:
                p1_two += 1
            elif board[i][j] == -1 and board[i+1][j+1] == -1 and board[i+2][j+2] == 0 and board[i+3][j+3] == 0:
                p2_two -= 1
            if board[i][j] == 1 and board[i+1][j+1] == 0 and board[i+2][j+2] == 1 and board[i+3][j+3] == 0:
                p1_two += 1
            elif board[i][j] == -1 and board[i+1][j+1] == 0 and board[i+2][j+2] == -1 and board[i+3][j+3] == 0:
                p2_two -= 1
            if board[i][j] == 0 and board[i+1][j+1] == 1 and board[i+2][j+2] == 1 and board[i+3][j+3] == 0:
                p1_two += 1
            elif board[i][j] == 0 and board[i+1][j+1] == -1 and board[i+2][j+2] == -1 and board[i+3][j+3] == 0:
                p2_two -= 1
            if board[i][j] == 0 and board[i+1][j+1] == 1 and board[i+2][j+2] == 0 and board[i+3][j+3] == 1:
                p1_two += 1
            elif board[i][j] == 0 and board[i+1][j+1] == -1 and board[i+2][j+2] == 0 and board[i+3][j+3] == -1:
                p2_two -= 1
            if board[i][j] == 0 and board[i+1][j+1] == 0 and board[i+2][j+2] == 1 and board[i+3][j+3] == 1:
                p1_two += 1
            elif board[i][j] == 0 and board[i+1][j+1] == 0 and board[i+2][j+2] == -1 and board[i+3][j+3] == -1:
                p2_two -= 1


def manhattan_distance(point1, point2):
    dist = 0
    for (dim1, dim2) in zip(point1, point2):
        dist = abs(dim1 - dim2)
    return dist

def euclidean_distance(point1, point2):
    dist = 0
    for (dim1, dim2) in zip(point1, point2):
        dist = (dim1 - dim2) ** 2
    return dist ** (1/2)
    
def get_mean_of_points(points):
    mean = []
    for dim in zip(*points):
        mean.append(sum(dim)/len(dim))
    return mean

def dist_heuristic(board):
    # Calculate the distance of each players piece from the mean of their pieces
    sse_vals = []
    for player in [-1, 1]:
        points = board.get_piece_locations(player)
        mean = get_mean_of_points(points)
        sse = 0
        for point in points:
            sse += manhattan_distance(point, mean) ** 2
        sse_vals.append(sse)
    # Closer to 1 = Better for player one
    return sse_vals[0] - sse_vals[1]

def get_line_length(board, line):
    line_length = 0
    player = board.get_space(line[0][0], line[0][1])
    stop = False
    for point in line:
        row, col = point
        # If any piece belongs to the other player than the row can't be completed
        # or if the spot isn't on the board
        if (not board.is_on_board(row, col)) or (board.get_space(row, col) == (player * -1)):
            stop = True
            line_length = 0
        elif board.is_on_board(row, col) and board.get_space(row, col) == player:
            if not stop: line_length += 1
        else:
            stop = True

    return line_length

def get_partial_lines(board, player):
    partial_lines = {}

    for line_length in range(2,board.x):
        partial_lines[line_length] = 0

    line_list = []
    for piece in board.get_piece_locations(player):
        row, col = piece
        cardinal_lines = [
            tuple((row, col - x) for x in range(board.x)),
            tuple((row, col + x) for x in range(board.x)),
            tuple((row - x, col) for x in range(board.x)),
            tuple((row + x, col) for x in range(board.x))
        ]
        diagonal_lines = [
            tuple((row - x, col - x) for x in range(board.x)),
            tuple((row - x, col + x) for x in range(board.x)),
            tuple((row - x, col + x) for x in range(board.x)),
            tuple((row + x, col + x) for x in range(board.x))
        ]

        line_list = line_list + cardinal_lines + diagonal_lines

    for line in set(line_list):
        #print(line)
        line_length = get_line_length(board, line)
        if line_length in partial_lines:
            partial_lines[line_length] += 1

    return partial_lines

def partial_lines_heuristic(board):
    # This heuristics counts the number of partial lines for each player
    partial_line_weights = []
    weight = 1
    weight_decay_rate = 1
    for index in range(2,board.x):
        partial_line_weights.insert(0, weight)
        weight *= (1 - weight_decay_rate)
    
    player_one_partial_lines = get_partial_lines(board, 1).values()
    player_two_partial_lines = get_partial_lines(board, -1).values()
    heuristic_value = 0
    for p1_lines, p2_lines, line_weight in zip(player_one_partial_lines, player_two_partial_lines, partial_line_weights):
        # print("p1_lines")
        # print(p1_lines)
        # print("p2_lines")
        # print(p2_lines)
        # print("line_weight")
        # print(line_weight)
        heuristic_value += (p1_lines - p2_lines) * line_weight

    # print(player_one_partial_lines)
    # print(player_two_partial_lines)
    # print(partial_line_weights)
    # print(heuristic_value)
    return heuristic_value

def table_heuristic(board):
    val_table = [
        [3,4,5,7,5,4,3],
        [4,6,8,10,8,6,4],
        [5,8,11,13,11,8,5],
        [5,8,11,13,11,8,5],
        [4,6,8,10,8,6,4],
        [3,4,5,7,5,4,3],
    ]
    val_array = np.array(val_table)
    weighted_array = np.array(board.to_array()) * val_array
    summed_arr = sum(sum(weighted_array))
    # print(weighted_array)
    # print(summed_arr)
    return summed_arr


def random_heuristic(board):
    return random.uniform(-0.999, 0.999)
