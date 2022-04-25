# Tests two Connect X Players against each other and compares their performance


from GUIBoard import GUIBoard


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
            winner = self.play_game(first_player=game%2)
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
        print(f'Results from {self.num_games} Games between {self.player_one_name} (Player 1) and {self.player_two_name} (Player 2)')
        print(f'Player 1 ({self.player_one_name}) Win Percentage: {player_one_win_percent * 100}%')
        print(f'Player 2 ({self.player_two_name}) Win Percentage: {player_two_win_percent * 100}%')
        print(f'Tie Percentage: {ties_percent * 100}%')
        if (player_one_wins > player_two_wins):
            print(f'Overall Winner: Player 1 ({self.player_one_name}) with Win % {player_one_win_percent * 100}')
        else:
            print(f'Overall Winner: Player 2 ({self.player_two_name}) with Win % {player_two_win_percent * 100}')
        return results, player_one_wins, player_two_wins, ties

    def play_game(self, first_player):
        # Clone the given board so that moves don't effect the base board
        test_board = self.connect_x_board.clone_board()
        while (test_board.winner == None):
            if (first_player == 0):
                if (self.render == 'pygame'):
                    GUIBoard(test_board).drawGUIboard()
                    print(test_board.to_string())
                if (self.pause):
                    input("(P2 Moved) Press Enter to continue...")
                player_one_move = self.player_one.get_move(test_board)
                print(f'(P1 Move) {player_one_move}')
                test_board = test_board.make_move(player_one_move, 1)
                if (test_board.winner == None):
                    if (self.render == 'pygame'):
                        GUIBoard(test_board).drawGUIboard()
                        print(test_board.to_string())
                    if (self.pause):
                        input("(P1 Moved) Press Enter to continue...")
                    player_two_move = self.player_two.get_move(test_board)
                    print(f'(P2 Move) {player_two_move}')
                    test_board = test_board.make_move(player_two_move, -1)
            else:
                if (self.render == 'pygame'):
                    GUIBoard(test_board).drawGUIboard()
                    print(test_board.to_string())
                if (self.pause):
                    input("(P1 Moved) Press Enter to continue...")
                player_two_move = self.player_two.get_move(test_board)
                print(f'(P2 Move) {player_two_move}')
                test_board = test_board.make_move(player_two_move, -1)
                if (test_board.winner == None):
                    if (self.render == 'pygame'):
                        GUIBoard(test_board).drawGUIboard()
                        print(test_board.to_string())
                    if (self.pause):
                        input("(P2 Moved) Press Enter to continue...")
                    player_one_move = self.player_one.get_move(test_board)
                    print(f'(P1 Move) {player_one_move}')
                    test_board = test_board.make_move(player_one_move, 1)
        print(test_board.to_string())
        return test_board.winner
        
