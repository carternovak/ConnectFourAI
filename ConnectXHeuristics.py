import random
from collections import Set

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

def random_heuristic(board):
    return random.uniform(-0.999, 0.999)
