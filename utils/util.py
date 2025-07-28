import os
import yaml
import matplotlib.pyplot as plt

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

def plot(qpos_history, target_history):
    qpos_array = np.array(qpos_history)     # Shape: (T, 14)
    target_array = np.array(target_history) # Shape: (T, 14)
    
    num_dims = qpos_array.shape[1]
    fig, axs = plt.subplots(num_dims, 1, figsize=(10, 2*num_dims), sharex=True)

    for i in range(num_dims):
        axs[i].plot(qpos_array[:, i], label='qpos', color='blue')
        axs[i].plot(target_array[:, i], label='target', color='orange', linestyle='--')
        axs[i].set_ylabel(f'Dim {i}')
        axs[i].legend(loc='upper right')
        axs[i].grid(True)

    axs[-1].set_xlabel('Time Step')
    plt.tight_layout()
    plt.show()