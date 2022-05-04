import torch
from pathlib import Path
from ConnectXDQNTrainer import DQNAgent
import json

def save_model(model, model_name):

    # Get the path of this file
    script_dir = Path(__file__).resolve().parent
    folder_path = f'{script_dir}/models/{model_name}'

    # Get data about the model that will be used to recreate it later
    model_data = {}
    model_data['width'] = model.board_width
    model_data['height'] = model.board_height
    model_data['conv'] = model.conv_model
    model_data['training_time'] = model.total_training_time

    # Create new folder to store data about the model being saved
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    # Save the model parameters
    torch.save(model.policy_net.state_dict(), Path(f'{folder_path}/policy.pt'))
    torch.save(model.target_net.state_dict(), Path(f'{folder_path}/target.pt'))

    # Save the model data
    with open(f'{folder_path}/model_metadata.json', 'w') as output_file:
        json.dump(model_data, output_file)

def load_model(model_name):
    script_dir = Path(__file__).resolve().parent
    folder_path = f'{script_dir}/models/{model_name}'

    # Load the metadata about the model
    with open(f'{folder_path}/model_metadata.json', 'r') as model_metadata_file:
        model_metadata = json.load(model_metadata_file)

    # Load the model itself
    loaded_model = DQNAgent(None, 
                            conv_model = model_metadata['conv'], 
                            board_width = model_metadata['width'], 
                            board_height = model_metadata['height'], 
                            action_states = model_metadata['width'], 
                            batch_size = 0, 
                            epsilon = 0, 
                            epsilon_decay = 0, 
                            min_epsilon=0, 
                            gamma = 0, 
                            lr = 0,
                            pre_trained_policy = torch.load(Path(f'{folder_path}/policy.pt')),
                            pre_trained_target = torch.load(Path(f'{folder_path}/target.pt')))

    return loaded_model