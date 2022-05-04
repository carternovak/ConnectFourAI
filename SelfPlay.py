import torch
from ConnectXDQNTrainer import DQNAgent
from ConnectXGym import ConnectXGym
from ConnectXBoard import ConnectXBoard
from RandomConnect4Player import RandomConnect4Player
from DQNConnect4Player import DQNConnect4Player
from AlphaBetaConnect4Player import AlphaBetaConnect4Player
from ConnectXHeuristics import table_heuristic, respective_powered
from ConnectXTester import ConnectXTester

def self_play(board, generations, episodes_per_generation = 5000, learning_episodes = 500):

    env = ConnectXGym(ConnectXBoard(), RandomConnect4Player(), 1, -1)

    starting_model = DQNAgent(env, board.width, board.height, board.width, conv_model = True, batch_size=128, lr=0.001, gamma = 0.5, epsilon=0.9, epsilon_decay=0.0001, min_epsilon=0.05)
    # Train the model against a random player for a few iterations

    starting_model.train(learning_episodes)

    best_model_generation = 0
    best_model = starting_model

    for generation in range(generations):

        new_env = ConnectXGym(ConnectXBoard(), DQNConnect4Player(best_model), 1, -1)
        new_model = DQNAgent(new_env, board.width, board.height, board.width, conv_model = True, batch_size = 128, lr = 0.001, gamma = 0.5, epsilon=0.9, epsilon_decay = 0.0001, min_epsilon = 0.05)
        new_model.train(episodes_per_generation)

        best_model.env.other_player = DQNConnect4Player(new_model)
        best_model.train(episodes_per_generation)

        # Compare the two models
        tester = ConnectXTester(DQNConnect4Player(best_model), "Old Model", DQNConnect4Player(new_model), "New Model", 500, ConnectXBoard(), render=None)
        _, p1_wins, p2_wins, _ = tester.test(False)
        # print(best_model == new_model)
        if (p1_wins > p2_wins):
            # Current model is better
            best_model = best_model
            print(f'Old Model Won with {p1_wins} wins')
            print(f'(Gen {generation + 1}) Current Best Model (Gen {best_model_generation}) Won with {p1_wins} wins')
        else:
            # print(f'New Model Won with {p2_wins} wins')
            # New model is better
            print(f'(Gen {generation + 1}) New Model (Gen {generation + 1}) Won with {p2_wins} wins')
            best_model = new_model
            best_model_generation = generation + 1

        print(f'Gen {generation + 1} done')
    # Repeat X times
    return best_model

def true_self_play(board, generations, episodes_per_generation = 5000, learning_episodes = 500):
    
        #rand_player = RandomConnect4Player()
        training_player = AlphaBetaConnect4Player(respective_powered, 1, 25)
        env = ConnectXGym(ConnectXBoard(), training_player, 1, -1)
        starting_model = DQNAgent(env, board.width, board.height, board.width, conv_model = True, batch_size=128, lr = 0.001, gamma = 0.5, epsilon=0.9, epsilon_decay = 0.001, min_epsilon = 0.05)
        starting_model.train(learning_episodes)
        competitor_model = DQNAgent(env, board.width, board.height, board.width, conv_model = True, batch_size=128, lr = 0.001, gamma = 0.5, epsilon=0.9, epsilon_decay = 0.001, min_epsilon = 0.05)
        competitor_model.policy_net.load_state_dict(starting_model.policy_net.state_dict())
        competitor_model.policy_net.load_state_dict(starting_model.policy_net.state_dict())
        starting_model.env.other_player = DQNConnect4Player(competitor_model)

        for generation in range(generations):
            starting_model.train(episodes_per_generation)
            competitor_model.policy_net.load_state_dict(starting_model.policy_net.state_dict())
            competitor_model.policy_net.load_state_dict(starting_model.policy_net.state_dict())
            print(f'Gen {generation + 1} done')
        return starting_model