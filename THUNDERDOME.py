from ConnectXGym import ConnectXGym
from DQNConnect4Player import DQNConnect4Player
import random
import time

class ThunderDome:

    # Takes two dicts of models to be tested
    def __init__(self, learning_models, non_learning_models, learning_model_bias = .3):
        self.learning_models = learning_models
        self.non_learning_models = non_learning_models
        self.learning_model_bias = learning_model_bias

    def thunderdome(self, bouts, episodes):
        for bout in range(bouts):
            # randomly select one learning model and one (possibly) learning model
            learning_model_name = random.choice(list(self.learning_models.keys()))
            learning_model = self.learning_models[learning_model_name]
            start_time = time.time()
            if (random.random() > self.learning_model_bias):
                other_model_name = random.choice(list(self.non_learning_models.keys()))
                other_model = self.non_learning_models[other_model_name]
                self.train_model_against_static_model(learning_model, other_model, episodes)
            else:
                other_model_name = random.choice(list(self.learning_models.keys()))
                other_model = self.learning_models[other_model_name]
                self.train_model_against_learning_model(learning_model, other_model, episodes)
            end_time = time.time()
            print(f'Bout {bout+1} {learning_model_name} vs {other_model_name} Complete in {round(end_time - start_time, 2)}s ({episodes} Episodes)')

    def train_model_against_static_model(self, model, non_learning_model, episodes):
        model.env.other_player = non_learning_model
        model.train(episodes)

    def train_model_against_learning_model(self, model, learning_model, episodes):
        model.env.other_player = DQNConnect4Player(learning_model)
        model.train(episodes)
        # Swap positions and train again
        learning_model.env.other_player = DQNConnect4Player(model)
        learning_model.train(episodes)

    def add_model(self, name, model, learning = True):
        if (learning):
            self.learning_models[name] = model
        else:
            self.non_learning_models[name] = model

    def get_trained_models(self):
        return self.learning_models
