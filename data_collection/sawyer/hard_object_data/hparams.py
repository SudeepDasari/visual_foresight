""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path
from visual_mpc.policy.random_policy import Randompolicy
from visual_mpc.agent.general_agent import GeneralAgent
from envs.sawyer_robot.autograsp_sawyer_env import AutograspSawyerEnv


if 'VMPC_DATA_DIR' in os.environ:
    BASE_DIR = os.path.join(os.environ['VMPC_DATA_DIR'], 'towel_pick/')
else:
    BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))


agent = {
    'type': GeneralAgent,
    'env': (AutograspSawyerEnv, {}),
    'data_save_dir': BASE_DIR,
    'T': 30,
    'image_height' : 240,
    'image_width' : 320,
    'record': BASE_DIR + '/record/',
}


policy = {
    'type': Randompolicy,
    'nactions': 10,
    'initial_std': 0.035,   #std dev. in xy
    'initial_std_lift': 0.08,   #std dev. in z
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