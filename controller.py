import torch
from utils.dataset import set_seed
import os
from model.policy import ControllerACTPolicy
import pickle
import time
import numpy as np
from einops import rearrange
import yaml
import argparse
from utils.misc import load_task_config
import matplotlib.pyplot as plt
from aloha.robot_utils import move_grippers
from aloha.real_env import make_real_env, make_real_env_and_spin

FPS = 50
PUPPET_GRIPPER_JOINT_OPEN = 1.4910


class SingleActionController():

    def __init__(self, model_config, task_config, env, ckpt_name='policy_best.ckpt'):
        set_seed(1000)
        # handle configuation merge
        ckpt_dir = os.path.expandvars(task_config['ckpt_dir'])
        self.camera_names = task_config.get('camera_names', model_config['camera_names'])
        self.max_timesteps = task_config['episode_length']
        self.temporal_agg = task_config['temporal_agg']
        chunk_size = task_config.get('chunk_size', model_config['chunk_size']) # num_queries
        for k, v in task_config.items():
            if k in model_config:
                model_config[k] = v

        # load policy
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        self.policy = ControllerACTPolicy(model_config)
        self.policy.deserialize(torch.load(ckpt_path))
        self.policy.cuda()
        self.policy.eval()

        # load stats
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        self.pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        self.post_process = lambda a: a * stats['action_std'] + stats['action_mean']

        # reference to env
        self.robot = env

        # config temporal aggregation
        if self.temporal_agg:
            self.query_frequency = 1
            self.num_queries = chunk_size
        else:
            self.query_frequency = chunk_size
            self.num_queries = None

        # set time to run
        self.max_timesteps = int(self.max_timesteps * 1) # may increase for real-world tasks
        self.DT = 1 / FPS

    def reset(self):
        return self.robot.reset()

    def run(self):
        # home the arms
        ts = self.reset()

        # create storage for history and aggregation
        if self.temporal_agg:
            all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps+self.num_queries, 16]).cuda()

        with torch.inference_mode():
            start_time = time.time()
            culmulated_delay = 0

            qpos_history = []
            target_qpos_history = []

            for t in range(self.max_timesteps):
                loop_time = time.time()
                
                # get q obs
                obs = ts.observation
                qpos_numpy = np.array(obs['qpos'])
                qpos = self.pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

                # warm up
                if t == 0:
                    curr_image = self.get_image(ts, self.camera_names)
                    for _ in range(10):
                        self.policy(qpos, curr_image)
                    loop_time = time.time()

                # query policy
                if t % self.query_frequency == 0:
                    curr_image = self.get_image(ts, self.camera_names)
                    all_actions = self.policy(qpos, curr_image)

                
                # assign action
                if self.temporal_agg:
                    all_time_actions[[t], t:t+self.num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % self.query_frequency]

                # post-process action
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = self.post_process(raw_action)
                target_qpos = action[:-2]
                base_action = action[-2:]
                
                # step the environment
                ts = self.robot.step(target_qpos, base_action)

                qpos_history.append(qpos_numpy)
                target_qpos_history.append(target_qpos)

                # keep pace
                duration = time.time() - loop_time
                sleep_time = max(0, self.DT - duration)
                time.sleep(sleep_time)

                # logging
                if duration >= self.DT:
                    culmulated_delay += (duration - self.DT)
                    print(f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: {self.DT} s, culmulated delay: {culmulated_delay:.3f} s')

            print(f'Avg fps {self.max_timesteps / (time.time() - start_time)}')
            plot(qpos_history, target_qpos_history)

    def get_image(self, ts, camera_names):
        curr_images = []
        for cam_name in camera_names:
            curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
            curr_images.append(curr_image)
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        return curr_image


    def open_grippers(self):
        move_grippers([self.robot.follower_bot_left, self.robot.follower_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, 0.5)


def dead_rekckoning_turn(robot):
    '''
    Turn right for 2 seconds at pi/6 rad/s for a 60 deg turn.
    We have no gyro :,) wtf
    '''
    arm_action, base_action = robot.get_qpos(), (0, -np.pi/6) # linear, angular
    robot.step(arm_action, base_action)
    time.sleep(2)
    robot.step(arm_action, (0, 0))


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

def main(args):
    with open('config/model_configs/default_config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)
    task_config = load_task_config(args['task_name'])

    robot = make_real_env_and_spin(setup_robots=True, setup_base=True)
    sac = SingleActionController(model_config, task_config, robot)
    sac.run()
    sac.open_grippers()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name', type=str, required=True, help='Name of the task (corresponding to task config YAML)')

    args = parser.parse_args()
    args = vars(args)

    main(args)

    # [0.003, -1.845, 1.615, -0.004, -1.917, -0.009, 1.33] 
    # [-0.009, -1.841, 1.624, 0, -1.908, 0.026, 1.322]

    # [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
    # [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

    # import threading

    # robot = make_real_env(setup_robots=True, setup_base=True)
    # robot.reset()

    # def print_qpos():
    #     while True:
    #         print(robot.get_qpos())
    # threading.Thread(target=print_qpos, daemon=True).start()
    # while True:
    #     move_grippers(
    #         [robot.follower_bot_left, robot.follower_bot_right],
    #         [PUPPET_GRIPPER_JOINT_OPEN] * 2,
    #         3
    #     )

    # ts = robot.reset()
    # q_init = ts.observation['qpos']
    # print(f'{q_init}, {len(q_init)}')

    # for i in range(60):
    #     print('looping')
    #     time.sleep(1)

    #     qpos_obs = ts.observation['qpos']
    #     qpos = qpos_obs + 0.00

    #     ts = robot.step(qpos)
    #     print(f"{ts.observation['is_set_left']}, {ts.observation['is_set_right']}")

    # q_final = ts.observation['qpos']

    # print(q_init - q_final)

    # q_init