""" Hyperparameters for Large Scale Data Collection (LSDC) """

import numpy as np
import os.path
from visual_mpc.policy.random.gaussian import GaussianPolicy
from visual_mpc.agent.general_agent import GeneralAgent
from visual_mpc.envs.robot_envs.autograsp_env import AutograspEnv
from visual_mpc.envs.robot_envs.util.topic_utils import IMTopic
from visual_mpc.policy.random.sampler_policy import SamplerPolicy

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))


env_params = {
    'email_login_creds': '.email_cred',
    'camera_topics': [
                      IMTopic('/front/image_raw'),
                      IMTopic('/left/image_raw', flip=True),
                      IMTopic('/right_side/image_raw'),
                      IMTopic('/left_side/image_raw'),
                      IMTopic('/right/image_raw')
    ],
}


agent = {
    'type': GeneralAgent,
    'env': (AutograspEnv, env_params),
    'data_save_dir': BASE_DIR,
    'T': 30,
    'image_height' : 240,
    'image_width' : 320,
    'record': BASE_DIR + '/record/',
}


policy = {
    'type': SamplerPolicy,
    'mean_bias': np.array([0.,0.,-0.04, 0.])
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
