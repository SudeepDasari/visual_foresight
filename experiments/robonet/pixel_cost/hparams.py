import numpy as np
import os
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.envs.sawyer_robot.autograsp_sawyer_env import AutograspSawyerEnv
from visual_mpc.policy.cem_controllers.pixel_cost_controller import PixelCostController
from visual_mpc.envs.sawyer_robot.util.topic_utils import IMTopic


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))


env_params = {
    'start_box': [1, 1, 0.7],
    'reset_before_eval': False,
    'rand_drop_reset': False,
    'save_video': True,
    'camera_topics': [IMTopic('/test/image_raw')]
}

agent = {'type' : BenchmarkAgent,
         'env': (AutograspSawyerEnv, env_params),
         'data_save_dir': BASE_DIR,
         'T': 13,  #number of commands per episodes (issued at control_rate / substeps HZ)
         'image_height': 48,
         'image_width': 64,
         'current_dir': current_dir,
         'make_final_recording': ''
}

policy = {
    'type': PixelCostController,
    'replan_interval': 13,
    'num_samples': 600,
    'selection_frac': 0.05,
    'predictor_propagation': True,   # use the model get the designated pixel for the next step!
    'initial_std_lift': 0.2,  # std dev. in xy
    'initial_std_rot': np.pi / 10,
    'rejection_sampling': False,
    'nactions': 13,
    'repeat': 1
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
