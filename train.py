import torch
import yaml
import numpy as np
import os
import pickle
import argparse
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from clearml import Task
from util.dataset import load_data # data functions
from util.dataset import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, postprocess_base_action # helper functions
from model.policy import ACTPolicy


def main(args):
    with open('config/model_configs/default_config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)
    task_config = load_task_config(args['task_name'])

    # override model
    for k, v in task_config.items():
        if k in model_config:
            model_config[k] = v
    model_config['lr'] = args['lr']
    model_config['weight_decay'] = args['weight_decay']
    model_config['kl_weight'] = args['kl_weight']


    dataset_dir = os.path.expandvars(task_config['dataset_dir'])

    training_config = {
        'num_steps': args['num_steps'],
        'validate_every': args['validate_every'],
        'save_every': args['save_every'],
        'ckpt_dir': args['ckpt_dir'],
        'lr': args['lr'],
        'model_config': model_config,
        'task_config': task_config,
        'task_name': args['task_name'],
        'seed': args['seed'],
        'camera_names': model_config['camera_names'],
        'episode_length': task_config['episode_length']
    }

    # cml_task = Task.init(
    #     project_name="mobile_aloha",
    #     task_name="training",
    #     task_type=Task.TaskTypes.training
    # )
    # cml_task.connect(training_config)

    train_dataloader, val_dataloader, stats, _ = \
        load_data(
            dataset_dir,
            lambda n: True,
            model_config['camera_names'],
            args['batch_size'],
            args['batch_size'],
            model_config['chunk_size'],
            True,
            False,
            'ACT',
            stats_dir_l=None,
            sample_weights=None,
            train_ratio=args['train_ratio']
        )
    
    stats_path = os.path.join(args['ckpt_dir'], f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # best_ckpt_info = train_bc(train_dataloader, val_dataloader, training_config, cml_task)
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, training_config, None)

    best_step, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(args['ckpt_dir'], f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}')
    

def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config, cml_task):
    num_steps = config['num_steps']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_config = config['model_config']
    validate_every = config['validate_every']
    save_every = config['save_every']

    set_seed(seed)

    policy = ACTPolicy(policy_config)
    policy.cuda()
    optimizer = policy.configure_optimizers()

    min_val_loss = np.inf
    best_ckpt_info = None
    
    train_dataloader = repeater(train_dataloader)
    for step in tqdm(range(num_steps+1)):
        # validation
        if step % validate_every == 0:
            print('validating')

            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    validation_dicts.append(forward_dict)
                    if batch_idx > 50:
                        break

                validation_summary = compute_dict_mean(validation_dicts)

                epoch_val_loss = validation_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))
            for k in list(validation_summary.keys()):
                validation_summary[f'val_{k}'] = validation_summary.pop(k)            
            # for k, v in validation_summary.items():
            #     cml_task.get_logger().report_scalar(
            #         title=k,
            #         series='val',
            #         value=v,
            #         iteration=step
            #     )

            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in validation_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        data = next(train_dataloader)
        forward_dict = forward_pass(data, policy)
        # backward
        loss = forward_dict['loss']
        loss.backward()
        optimizer.step()
        # for k, v in forward_dict.items():
        #     cml_task.get_logger().report_scalar(
        #         title=k,
        #         series='train',
        #         value=v,
        #         iteration=step
        #     )

        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_step_{step}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.serialize(), ckpt_path)

    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_step_{best_step}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at step {best_step}')

    return best_ckpt_info


def load_task_config(name):
    directory = 'config/task_configs'
    filename = f'{name}.yaml'
    filepath = os.path.join(directory, filename)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Config file '{filename}' not found in '{directory}'")

    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    
    return config



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name', type=str, required=True, help='Name of the task (corresponding to task config YAML)')
    parser.add_argument('--num_steps', type=int, required=True, help='Number of training steps')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training/validation')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--kl_weight', type=int, required=True, help='')

    parser.add_argument('--weight_decay', type=int, default=1e-4, help='')
    parser.add_argument('--validate_every', type=int, default=500, help='Validation frequency')
    parser.add_argument('--save_every', type=int, default=1000, help='Checkpoint saving frequency')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--train_ratio', type=float, default=0.95, help='Train/val split ratio')

    args = parser.parse_args()
    args = vars(args)

    main(args)
