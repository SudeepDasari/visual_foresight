""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path
from visual_mpc.policy.random.gaussian import GaussianAGEpsilonPolicy
from visual_mpc.agent.general_agent import GeneralAgent
from visual_mpc.envs.robot_envs.vanilla_env import VanillaEnv
from visual_mpc.envs.robot_envs.util.topic_utils import IMTopic

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))


env_params = {
    'email_login_creds': '.email_cred',
    'camera_topics': [IMTopic('/front/image_raw', flip=True),
                      IMTopic('/left/image_raw'),
                      IMTopic('/right_side/image_raw'),
                      IMTopic('/left_side/image_raw'),
                      IMTopic('/right/image_raw')],
}


agent = {
    'type': GeneralAgent,
    'env': (VanillaEnv, env_params),
    'data_save_dir': BASE_DIR,
    'T': 30,
    'image_height' : 240,
    'image_width' : 320,
    'record': BASE_DIR + '/record/',
}


policy = {
    'type': GaussianAGEpsilonPolicy,
    'nactions': 30,
    'repeat': 1,
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
    'ngroup': 1000,
    'mode': 'train_eps'
}
