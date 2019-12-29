""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path
import numpy as np
from visual_mpc.policy.random.random_fold_policy import RandomFoldPolicy
from visual_mpc.agent.general_agent import GeneralAgent
from visual_mpc.envs.sawyer_robot.autograsp_sawyer_env import AutograspSawyerEnv
from visual_mpc.envs.sawyer_robot.util.topic_utils import IMTopic


if 'VMPC_DATA_DIR' in os.environ:
    BASE_DIR = os.path.join(os.environ['VMPC_DATA_DIR'], 'towel_pick/')
else:
    BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

env_params = {
    'lower_bound_delta': [0, 0., -0.01, 265 * np.pi / 180 - np.pi/2, 0],
    'upper_bound_delta': [0, -0.15, -0.01, 0., 0],
    'normalize_actions': True,
    'gripper_joint_thresh': 0.999856,
    'rand_drop_reset': False,
    'camera_topics': [IMTopic('/front/image_raw', flip=True),
                      IMTopic('/left/image_raw'),
                      IMTopic('/right_side/image_raw'),
                      IMTopic('/left_side/image_raw'),
                      IMTopic('/right/image_raw')],
    'zthresh':0.05   # gripper only closes very close to ground
}
agent = {
    'type': GeneralAgent,
    'env': (AutograspSawyerEnv, env_params),
    'data_save_dir': BASE_DIR,
    'T': 21,
    'image_height' : 240,
    'image_width' : 320
}

policy = {
    'type': RandomFoldPolicy
}

config = {
    'traj_per_file':128,
    'current_dir' : current_dir,
    'save_data': True,
    'save_raw_images': True,
    'start_index':0,
    'end_index': 120000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}