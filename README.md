# ConnectFourAI

This repository contains the code for our Deep Q-Learning and MiniMax(with Alpha Beta pruning) based Connect 4 AIs.

## Dependencies
Pytorch: Can be installed using `conda install -c pytorch pytorch` or `conda install -c conda-forge gym`

Numpy: Can be installed using `conda install -c conda-forge gym`

Open AI Gym: Can be installed with `conda install -c anaconda numpy`

Pygame: Can be installed with `conda install -c cogsci pygame`


## Sections

### Players:
These are the different classes that are used to play Connect 4. Each one uses a different method to determine which move to make
- RandomPlayer: makes moves randomly
- MiniMaxPlayer: use minimax with a heuristic to make moves
- AlphaBetaPlayer: use minimax with Alpha Beta pruning and a heuristic to make moves
- DQNPlayer: uses a DQN model to select moves
- HumanPlayer: make moves based on console input (used for testing)
- GreedPlayer: Always put a piece in the first available space

### Board (ConnectXBoard.py):
Store the state of the Connect4 board and contains functions used to get and set its state.

### GUI (GUIBoard.py):
This class contains the code used to render the board visually with Pygame

### Game (Connect4Game.py):
This contains the code used to allow a player to player against of the Connect 4 Ais

### Heuristics (ConnectXHeuristics.py):
This file contains the heuristics that are used by the Minimax and Minimax with Alpha Beta players to select moves

### Model Loading (ModelLoading.py):
This file contains the code used to save a DQN model, or load a saved DQN model to play against

### Deep Q-Learning (ConnectXDQNTrainer.py)
This file contains the implementation of the DQN we used, aswell as functions used to train it

### Training (TestDQN.py):
This is the file where the training was run for the different DQNs

### Testing (ConnectXTester.py):
This file contains functions used to test one model agaisnt another model or many models against each other


## How to Play

Run the `Connect4Game.py` file. Then input the name of the player you want to play against (Table, RP, or DQN) into the terminal.



