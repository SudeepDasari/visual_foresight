""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.policy.random.gaussian import GaussianPolicy
from visual_mpc.agent.general_agent import GeneralAgent
from visual_mpc.envs.mujoco_env.cartgripper_env.cartgripper_pusher import CartgripperPusherEnv
import numpy as np

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))


env_params = {
    'num_objects': 12,
    'object_mass': 0.5,
    'friction': 1.0,
    'minlen': 0.03,
    'maxlen': 0.06,
    'object_object_mindist': 0.15,
    'cube_objects': True,
    'ncam': 2
}

agent = {
    'type': GeneralAgent,
    'env': (CartgripperPusherEnv, env_params),
    'data_save_dir': BASE_DIR,
    'T': 30,
    'image_height' : 48,
    'image_width' : 64,
    'gen_xml': 400,   #generate xml every nth trajecotry
#    'rejection_sample': 1,
#    'make_final_gif': ''
}

policy = {
    'type' : GaussianPolicy,
    'nactions' : 10,
    'initial_std': 0.04,   #std dev. in xy
    'initial_std_lift': 0.6,   #std dev. in xy
    'initial_std_rot': np.pi / 32,
}

config = {
    'traj_per_file':64,
    'current_dir' : current_dir,
    'save_data': True,
    'seperate_good': False,
    'save_raw_images' : False,
    'start_index':0,
    'end_index': 60000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}
