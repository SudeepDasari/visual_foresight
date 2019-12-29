""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.policy.random.gaussian import GaussianPolicy
from visual_mpc.agent.general_agent import GeneralAgent
from visual_mpc.envs.sawyer_robot.autograsp_sawyer_env import AutograspSawyerEnv
from visual_mpc.envs.sawyer_robot.util.topic_utils import IMTopic

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))


conditional_override_nordri = {
    'env_params': {
        'camera_topics': [IMTopic('/kinect2/hd/image_color', left=150, right=200, bot=250, flip=True),
                          IMTopic('/camera/image_raw')],
        'gripper_attached': False
    }
}

conditional_override_nordri_swap = {
    'env_params': {
        'camera_topics': [IMTopic('/kinect2/hd/image_color', left=150, right=200, bot=250, flip=True),
                          IMTopic('/camera/image_raw')],
        'upper_bound_delta': [0., 0., -0.036, 0., 0.],
        'lower_bound_delta': [0., 0., -0.036, 0., 0.],
    }
}

conditional_override_vestri = {
    'env_params': {
        'upper_bound_delta': [0., 0., 0.047, 0., 0.],
        'lower_bound_delta': [0., 0., 0.047, 0., 0.],
        'gripper_attached': False
    }
}

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
    'type': GaussianPolicy,
    'nactions': 10,
    'initial_std': 0.035,   #std dev. in xy
    'initial_std_lift': 0.08,   #std dev. in z
}


config = {
    'traj_per_file':128,
    'current_dir' : current_dir,
    'override_nordri': conditional_override_nordri_swap,
    'override_vestri': conditional_override_vestri,
    'save_data': True,
    'save_raw_images': True,
    'start_index':0,
    'end_index': 120000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000,
    'mode': 'test'
}
