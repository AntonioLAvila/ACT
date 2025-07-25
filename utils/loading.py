import os
import yaml

def load_task_config(name):
    directory = 'config/task_configs'
    filename = f'{name}.yaml'
    filepath = os.path.join(directory, filename)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Config file '{filename}' not found in '{directory}'")

    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def load_model_config(name):
    directory = 'config/model_configs'
    filename = f'{name}.yaml'
    filepath = os.path.join(directory, filename)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Config file '{filename}' not found in '{directory}'")

    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    
    return config