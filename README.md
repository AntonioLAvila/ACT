# ACT

This repo contains the minimum to run and train act on Mobile Aloha. To run on the robot, it interfaces with [Aloha](https://github.com/AntonioLAvila/aloha) which interfaces with ROS/hardware.

## Structure:

Task configs and model configs are the backbone of the code. They are located in `config/task_configs` and `config/model_configs`. To train and run you need to create the relevant task config and model config. Keep in mind that anything specified in a task config will **override** the model config (this was done to make changing `chunk_size` easier). Also, the `ckpt_dir` value in task configs are only used when running the controller and **do not** affect save location when training. Task configs should align with recording task configs specified in [Aloha](https://github.com/AntonioLAvila/aloha).

## Training:

Once you've collected data and have a model and task config. `dataset_dir` from your task config is used to load episodes. You can train by running `train.py`. Read the main function in `train.py` to see all args. Example: `python3 train.py --task_name pick_up_wafer --ckpt_dir ~/ckpts/pick_up_wafer --num_steps 30000 --batch_size 64 --lr 5e-5 --kl_weight 30`.

## Running:

Once you have a policy you'd like to run. Specify the location of the checkpoints in your task config. By default the controller loads `policy_best.ckpt` in the checkpoint directory. To test the controller, run `pyhthon3 controller.py --task_name <your task name>`. Setting `base_only` in a task config only executes the base motions of a policy (however there will be issues if you include camera data since the arms won't be moving in the way the policy would expect). Only set it if you know what you're doing.
