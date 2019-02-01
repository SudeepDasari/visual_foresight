""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.policy.random.gaussian import GaussianPolicy
from visual_mpc.agent.general_agent import GeneralAgent
from visual_mpc.envs.mujoco_env.cartgripper_env.cartgripper_xz_grasp import CartgripperXZGrasp
import numpy as np

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))


env_params = {
    # resolution sufficient for 16x anti-aliasing
    'viewer_image_height': 96,
    'viewer_image_width': 128,
    'cube_objects': True
}


agent = {
    'type': GeneralAgent,
    'env': (CartgripperXZGrasp, env_params),
    'data_save_dir': BASE_DIR,
    'T': 30,
    'image_height' : 48,
    'image_width' : 64,
    'gen_xml': 1,   #generate xml every nth trajecotry
#    'make_final_gif': '',
    'rejection_sample': 5
}

policy = {
    'type' : GaussianPolicy,
    'nactions': 10,
    'action_order': ['x', 'z', 'grasp'],
    'initial_std_lift': 0.1,  # std dev. in xy
}

config = {
    'traj_per_file':128,
    'current_dir' : current_dir,
    'save_data': True,
    'seperate_good': True,
    'save_raw_images' : False,
    'start_index':0,
    'end_index': 100000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}
