import torch
from ConnectXDQNTrainer import DQNAgent
from ConnectXGym import ConnectXGym
from ConnectXBoard import ConnectXBoard
from RandomConnect4Player import RandomConnect4Player
from DQNConnect4Player import DQNConnect4Player
from AlphaBetaConnect4Player import AlphaBetaConnect4Player
from ConnectXHeuristics import table_heuristic
from ConnectXTester import ConnectXTester

def self_play(board, generations, episodes_per_generation = 5000, learning_episodes = 500):

    # practice_agent = DQNAgent(ConnectXGym(ConnectXBoard(), None, 1, -1), 
    #                         conv_model=True, 
    #                         board_width=board.width, 
    #                         board_height=board.height, 
    #                         action_states=board.width, 
    #                         batch_size=100, 
    #                         epsilon=.999, 
    #                         epsilon_decay=0.01, 
    #                         min_epsilon=0.05, 
    #                         gamma=.5, 
    #                         lr=0.0001,
    #                         pre_trained_policy=torch.load('1-self-play-policy.pt'),
    #                         pre_trained_target=torch.load('1-self-play-target.pt'))
                        
    env = ConnectXGym(ConnectXBoard(), AlphaBetaConnect4Player(table_heuristic, -1, 125), 1, -1)

    starting_model = DQNAgent(env, board.width, board.height, board.width, conv_model = True, batch_size=128, lr=0.001, gamma = 0.5, epsilon=0.9, epsilon_decay=0.0001, min_epsilon=0.05)
    # Train the model against a random player for a few iterations

    starting_model.train(learning_episodes)

    best_model_generation = 0
    best_model = starting_model

    for generation in range(generations):

        # Create two new models, train them against the existing one and compare their results
        new_env = ConnectXGym(ConnectXBoard(), DQNConnect4Player(best_model), 1, -1)
        new_model = DQNAgent(new_env, board.width, board.height, board.width, conv_model = True, batch_size = 128, lr = 0.005, gamma = 0.5, epsilon=0.9, epsilon_decay = 0.0001, min_epsilon = 0.05)
        new_model.train(episodes_per_generation)

        # new_env2 = ConnectXGym(ConnectXBoard(), DQNConnect4Player(best_model), 1, -1)
        # new_model2 = DQNAgent(new_env2, board.width, board.height, board.width, conv_model = False, batch_size=128, lr=0.005, gamma=0.9, epsilon=0.9, epsilon_decay=0.0001, min_epsilon=0.05)
        # new_model2.train(episodes_per_generation)

        # best_model.env.other_player = DQNConnect4Player(new_model)
        # best_model.train(episodes_per_generation)
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
    
        # rand_model = RandomConnect4Player()
        env = ConnectXGym(ConnectXBoard(), AlphaBetaConnect4Player(table_heuristic, -1, 25), 1, -1)
        starting_model = DQNAgent(env, board.width, board.height, board.width, conv_model = True, batch_size=128, lr = 0.001, gamma = 0.9, epsilon=0.9, epsilon_decay = 0.001, min_epsilon = 0.05)
        starting_model.train(learning_episodes)
        competitor_model = DQNAgent(env, board.width, board.height, board.width, conv_model = True, batch_size=128, lr = 0.001, gamma = 0.9, epsilon=0.9, epsilon_decay = 0.001, min_epsilon = 0.05)
        competitor_model.policy_net.load_state_dict(starting_model.policy_net.state_dict())
        competitor_model.policy_net.load_state_dict(starting_model.policy_net.state_dict())
        starting_model.env.other_player = DQNConnect4Player(competitor_model)

        for generation in range(generations):
            starting_model.train(episodes_per_generation)
            competitor_model.policy_net.load_state_dict(starting_model.policy_net.state_dict())
            competitor_model.policy_net.load_state_dict(starting_model.policy_net.state_dict())
            print(f'Gen {generation + 1} done')
        return starting_model