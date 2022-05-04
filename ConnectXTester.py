# Tests two Connect X Players against each other and compares their performance


from ConnectXHeuristics import respective_powered, table_heuristic
from GUIBoard import GUIBoard
from DQNConnect4Player import DQNConnect4Player
from RandomConnect4Player import RandomConnect4Player
from MinMaxConnect4Player import MinMaxConnect4Player
from AlphaBetaConnect4Player import AlphaBetaConnect4Player
from ModelLoading import load_model
from ConnectXBoard import ConnectXBoard

import pandas as pd


class ConnectXTester():
    def __init__(self, player_one, player_one_name, player_two, player_two_name, num_games, connect_x_board, pause = False, render = None) -> None:
        self.player_one = player_one
        self.player_one_name = player_one_name
        self.player_two = player_two
        self.player_two_name = player_two_name
        self.num_games = num_games
        self.connect_x_board = connect_x_board
        self.pause = pause
        self.render = render

    def test(self, print_results = True):
        results = []
        for game in range(self.num_games):
            winner = self.play_game(first_player=game%2, print_results = print_results)
            results.append(winner)
            if (print_results):
                if (winner == 0):
                    print(f'Game {game+1} was a tie')
                if (winner == 1):
                    print(f'Player 1 ({self.player_one_name}) won Game {game+1}')
                if (winner == -1):
                    print(f'Player 2 ({self.player_two_name}) won Game {game+1}')
        player_one_wins = len([x for x in results if x == 1])
        player_two_wins = len([x for x in results if x == -1])
        ties = len([x for x in results if x == 0])
        player_one_win_percent = player_one_wins/len(results)
        player_two_win_percent = player_two_wins/len(results)
        ties_percent = ties/len(results)
        if (print_results):
            print(f'Results from {self.num_games} Games between {self.player_one_name} (Player 1) and {self.player_two_name} (Player 2)')
            print(f'Player 1 ({self.player_one_name}) Win Percentage: {player_one_win_percent * 100}%')
            print(f'Player 2 ({self.player_two_name}) Win Percentage: {player_two_win_percent * 100}%')
            print(f'Tie Percentage: {ties_percent * 100}%')
            if (player_one_wins > player_two_wins):
                print(f'Overall Winner: Player 1 ({self.player_one_name}) with Win % {player_one_win_percent * 100}')
            else:
                print(f'Overall Winner: Player 2 ({self.player_two_name}) with Win % {player_two_win_percent * 100}')
        return results, player_one_wins, player_two_wins, ties

    def play_game(self, first_player, print_results = False):
        # Clone the given board so that moves don't effect the base board
        test_board = self.connect_x_board.clone_board()
        while (test_board.winner == None):
            if (first_player == 0):
                if (self.render == 'pygame'):
                    GUIBoard(test_board).drawGUIboard()
                    print(test_board.to_string())
                if (self.pause):
                    input("(P2 Moved) Press Enter to continue...")
                player_one_move = self.player_one.get_move(test_board, 1)
                test_board = test_board.make_move(player_one_move, 1)
                if (test_board.winner == None):
                    if (self.render == 'pygame'):
                        GUIBoard(test_board).drawGUIboard()
                        print(test_board.to_string())
                    if (self.pause):
                        input("(P1 Moved) Press Enter to continue...")
                    player_two_move = self.player_two.get_move(test_board, -1)
                    test_board = test_board.make_move(player_two_move, -1)
            else:
                if (self.render == 'pygame'):
                    GUIBoard(test_board).drawGUIboard()
                    print(test_board.to_string())
                if (self.pause):
                    input("(P1 Moved) Press Enter to continue...")
                player_two_move = self.player_two.get_move(test_board, -1)
                test_board = test_board.make_move(player_two_move, -1)
                if (test_board.winner == None):
                    if (self.render == 'pygame'):
                        GUIBoard(test_board).drawGUIboard()
                        print(test_board.to_string())
                    if (self.pause):
                        input("(P2 Moved) Press Enter to continue...")
                    player_one_move = self.player_one.get_move(test_board, 1)
                    test_board = test_board.make_move(player_one_move, 1)
        if (print_results): print(test_board.to_string())
        return test_board.winner


def test_multiple_models(dict_of_players, board, num_games = 500):
    player_names = list(dict_of_players.keys())
    win_rates = {player_name: {other_player_name:None for other_player_name in player_names} for player_name in player_names}

    for player_name, player in dict_of_players.items():
        for other_player_name, other_player in dict_of_players.items():
            # Only test combinations that haven't been tested yet
            if (win_rates[player_name][other_player_name] == None):
                tester = ConnectXTester(player, player_name, other_player, other_player_name, num_games, board, pause = False, render = 'none')
                results, player_one_wins, player_two_wins, ties = tester.test(False)

                print(f'Test: {player_name} vs {other_player_name} Complete')

                win_rates[player_name][other_player_name] = player_one_wins
                if (player_name != other_player_name): 
                    win_rates[other_player_name][player_name] = player_two_wins
            else:
                print('skip')

    return win_rates

if __name__ == '__main__':
    models = {
        'DQN Random': DQNConnect4Player(load_model('15000-eps-random')),
        'DQN Table': DQNConnect4Player(load_model('15000-eps-table-25-depth')),
        'DQN Respective Power (Deep) 10,000': DQNConnect4Player(load_model('10000-eps-respective_powered-125-depth')),
        'DQN Respective Power 10,000': DQNConnect4Player(load_model('10000-eps-respective_powered-25-depth')),
        'DQN Respective Power 5,000': DQNConnect4Player(load_model('5000-eps-respective_powered-25-depth')),
        'DQN Respective Power 1,000': DQNConnect4Player(load_model('1000-eps-respective_powered-25-depth')),
        'Alpha Beta Table (125)': AlphaBetaConnect4Player(table_heuristic, -1, 125),
        'Alpha Beta Respective Power (125)': AlphaBetaConnect4Player(respective_powered, -1, 125),
        'Alpha Beta Table (50)': AlphaBetaConnect4Player(table_heuristic, -1, 50),
        'Alpha Beta Respective Power (50)': AlphaBetaConnect4Player(respective_powered, -1, 50),
        'Alpha Beta Table (5)': AlphaBetaConnect4Player(table_heuristic, -1, 5),
        'Alpha Beta Respective Power (5)': AlphaBetaConnect4Player(respective_powered, -1, 5),
        'Random': RandomConnect4Player(),
    }

    height = 6
    width = 7
    x = 4
    state_size = height * width
    action_size = width
    board = ConnectXBoard(height=height, width=width, x=x)

    results = test_multiple_models(models, board)

    results_df = pd.DataFrame(results)
    results_df.to_csv('results.csv')
    print(results_df)