""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path
from visual_mpc.policy.random.gaussian import GaussianPolicy
from visual_mpc.agent.general_agent import GeneralAgent
from visual_mpc.envs.robot_envs.autograsp_env import AutograspEnv
from visual_mpc.envs.robot_envs.util.topic_utils import IMTopic

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))


env_params = {
    'email_login_creds': '.email_cred',
    'camera_topics': [IMTopic('/front/image_raw'),
                      IMTopic('/left/image_raw'),
                      IMTopic('/right/image_raw')],
     #                 IMTopic('/left_side/image_raw'),
     #                 IMTopic('/right/image_raw')],
     'robot_type': 'widowx',
     'gripper_attached': 'default',
     'OFFSET_TOL': 3,
     'robot_upside_down': True,
     'zthresh': 0.2,
     'gripper_joint_thresh': 0.85,
     'gripper_joint_grasp_min': -0.9
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
    'type': GaussianPolicy,
    'nactions': 30,
    'repeat': 1,
    'initial_std': 0.035,   #std dev. in xy
    'initial_std_lift': 0.05,   #std dev. in z
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
