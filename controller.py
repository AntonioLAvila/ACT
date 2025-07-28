import torch
from utils.dataset import set_seed
import os
from model.policy import ControllerACTPolicy
import pickle
import time
import numpy as np
from einops import rearrange
import argparse
from utils.util import load_task_config, load_model_config, plot
from aloha.robot_utils import move_grippers
from aloha.real_env import make_real_env, make_real_env_and_spin

FPS = 50
PUPPET_GRIPPER_JOINT_OPEN = 1.4910


class SingleActionController():
    '''
    Holds the code for loading and rolling out a single policy once.
    NOTE: that `max_timesteps` is tunable. If the robot can't do the task
    in the specified time, increase this.
    '''
    def __init__(self, model_config, task_config, env, ckpt_name='policy_best.ckpt'):
        set_seed(1000)
        # handle configuation merge
        ckpt_dir = os.path.expandvars(task_config['ckpt_dir'])
        self.camera_names = task_config.get('camera_names', model_config['camera_names'])
        self.max_timesteps = task_config['episode_length']
        self.temporal_agg = task_config['temporal_agg']
        self.base_only = task_config['base_only']
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
                if self.base_only:
                    ts = self.robot.step_no_reqs(base_action=base_action)
                else:
                    ts = self.robot.step_no_reqs(action=target_qpos, base_action=base_action)

                # logging
                qpos_history.append(qpos_numpy)
                target_qpos_history.append(target_qpos)

                # keep pace
                duration = time.time() - loop_time
                sleep_time = max(0, self.DT - duration)
                time.sleep(sleep_time)

                # more logging
                if duration >= self.DT:
                    culmulated_delay += (duration - self.DT)
                    print(f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: {self.DT} s, culmulated delay: {culmulated_delay:.3f} s')

            print(f'Avg fps {self.max_timesteps / (time.time() - start_time)}')
            # plot(qpos_history, target_qpos_history)

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
    base_action = (0, -np.pi/6) # linear, angular
    robot.step_no_reqs(base_action=base_action)
    time.sleep(2)
    robot.step_no_reqs(base_action=(0, 0))


def main(args):
    model_config = load_model_config('default_config')
    task_config = load_task_config(args['task_name'])

    robot = make_real_env_and_spin(setup_robots=True, setup_base=True)

    sac = SingleActionController(model_config, task_config, robot, 'policy_step_40000_seed_0.ckpt')
    sac.run()
    sac.open_grippers()

    # dead_rekckoning_turn(robot)

    robot.shutdown()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name', type=str, required=False, help='Name of the task (corresponding to task config YAML)')

    args = parser.parse_args()
    args = vars(args)

    main(args)
