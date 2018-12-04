""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
import numpy as np
from visual_mpc.policy.random.gaussian import GaussianPolicy
from visual_mpc.agent.general_agent import GeneralAgent
from envs.mujoco_env.cartgripper_env.autograsp_env import AutograspCartgripperEnv


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))


env_params = {
    'num_objects': 4,
    'object_mass': 0.5,
    'friction': 1.0,
    'finger_sensors': True,
    'autograsp': {'zthresh': -0.06, 'touchthresh': 0.0, 'reopen': True}
}

agent = {
    'type': GeneralAgent,
    'env': (AutograspCartgripperEnv, env_params),
    'data_save_dir': BASE_DIR,
    'T': 30,
    'image_height' : 48,
    'image_width' : 64,
    'gen_xml': 400,   #generate xml every nth trajecotry
}

policy = {
    'type' : GaussianPolicy,
    'nactions' : 10,
    'initial_std': 0.02,   #std dev. in xy
    'initial_std_lift': 0.6,   #std dev. in xy
}

config = {
    'traj_per_file':128,
    'current_dir' : current_dir,
    'save_data': True,
    'save_raw_images' : False,
    'start_index':0,
    'end_index': 60000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}
