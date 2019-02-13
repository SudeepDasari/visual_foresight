""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path
from visual_mpc.policy.cem_controllers.variants.ensemble_vidpred import CEM_Controller_Ensemble_Vidpred
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.envs.mujoco_env.cartgripper_env.autograsp_env import AutograspCartgripperEnv
import numpy as np

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))


env_params = {
    'num_objects': 1,
    'object_mass': 0.5,
    'friction': 1.0,
    'finger_sensors': True,
    'minlen': 0.03,
    'maxlen': 0.06,
    'object_object_mindist': 0.15,
    'cube_objects': True,
    'autograsp': {'zthresh': -0.06, 'touchthresh': 0.0, 'reopen': True}
}


agent = {
    'type': BenchmarkAgent,
    'env': (AutograspCartgripperEnv, env_params),
    'T': 30,
    'image_height' : 48,
    'image_width' : 64,
    'data_save_dir': BASE_DIR,
    'make_final_gif_pointoverlay': True,
    'record': BASE_DIR + '/record/',
    'num_load_steps': 16,
    'start_goal_confs': os.environ['VMPC_DATA_DIR'] + '/ensemble_lifting_tasks',
    'current_dir': current_dir
}

policy = {
    'verbose':True,
    'initial_std': 0.04,  # std dev. in xy
    'initial_std_lift': 0.6,  # std dev. in xy
    'initial_std_rot': np.pi / 32,
    'type': CEM_Controller_Ensemble_Vidpred,
    'rejection_sampling': False,
    'replan_interval': 10,
    'num_samples': [800, 400],
}

config = {
    'current_dir': current_dir,
    'save_data': True,
    'save_raw_images': True,
    'start_index':0,
    'end_index': 88,
    'agent': agent,
    'policy': policy,
}
