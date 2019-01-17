""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.policy.random.gaussian import GaussianPolicy
from visual_mpc.agent.general_agent import GeneralAgent
from visual_mpc.envs.mujoco_env.sawyer_env.base_sawyer_env import SawyerEnv
import numpy as np

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))


env_params = {
    'num_objects': 6
}

agent = {
    'type': GeneralAgent,
    'env': (SawyerEnv , env_params),
    'data_save_dir': BASE_DIR,
    'T': 30,
    'image_height' : 48,
    'image_width' : 64,
    'gen_xml': 400,   #generate xml every nth trajecotry
    'make_final_gif': ''
}

policy = {
    'type' : GaussianPolicy,
    'nactions' : 10,
    'initial_std': 0.04,   #std dev. in xy
    'initial_std_lift': 0.6,   #std dev. in xy
#    'initial_std_rot': np.pi / 8,
}

config = {
    'traj_per_file':128,
    'current_dir' : current_dir,
    'save_data': True,
    'seperate_good': True,
    'save_raw_images' : True,
    'start_index':30000,
    'end_index': 60000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}
