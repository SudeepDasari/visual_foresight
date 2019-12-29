import numpy as np
import os
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.envs.sawyer_robot.autograsp_sawyer_env import AutograspSawyerEnv
from visual_mpc.policy.cem_controllers.human_cem_controller import HumanCEMController
from visual_mpc.envs.sawyer_robot.util.topic_utils import IMTopic


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))


env_params = {
    'lower_bound_delta': [0, 0., -0.01, 265 * np.pi / 180 - np.pi/2, 0],
    'upper_bound_delta': [0, -0.15, -0.01, 0., 0],
    'start_box': [1, 1, 0.7],
    'normalize_actions': True,
    'gripper_joint_thresh': 0.999856,
    'reset_before_eval': False,
    'rand_drop_reset': False,
    'save_video': True,
    'camera_topics': [IMTopic('/front/image_raw', flip=True),
                      IMTopic('/left/image_raw')]
}

agent = {'type' : BenchmarkAgent,
         'env': (AutograspSawyerEnv, env_params),
         'data_save_dir': BASE_DIR,
         'T': 30,  #number of commands per episodes (issued at control_rate / substeps HZ)
         'image_height': 48,
         'image_width': 64,
         'current_dir': current_dir,
         'make_final_recording': '',
         'no_goal_def': ''
}

policy = {
    'type': HumanCEMController,
    'replan_interval': 10,
    'num_samples': 50,
    'selection_frac': 0.1,
    'initial_std_lift': 0.2,  # std dev. in xy
    'initial_std_rot': np.pi / 10,
    'rejection_sampling': False,
    'state_append': [0.41, 0.25, 0.166]
}

config = {
    'traj_per_file':128,
    'current_dir' : current_dir,
    'save_data': True,
    'save_raw_images' : True,
    'start_index':0,
    'end_index': 30000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000,
    'nshuffle' : 200
}
