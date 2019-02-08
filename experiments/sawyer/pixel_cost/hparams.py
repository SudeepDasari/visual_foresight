import numpy as np
import os
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.envs.sawyer_robot.autograsp_sawyer_env import AutograspSawyerEnv
from visual_mpc.policy.cem_controllers.pixel_cost_controller import PixelCostController
from visual_mpc.envs.sawyer_robot.util.topic_utils import IMTopic


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))


env_params = {
    'reset_before_eval': False,
    'video_save_dir':  '',   # doesn't matter what you put here (should make this more elegant)
    'camera_topics': [IMTopic('/front/image_raw', flip=True),
                      IMTopic('/left/image_raw')]
}

agent = {'type' : BenchmarkAgent,
         'env': (AutograspSawyerEnv, env_params),
         'data_save_dir': BASE_DIR,
         'T': 15,  #number of commands per episodes (issued at control_rate / substeps HZ)
         'image_height': 48,
         'image_width': 64,
         'current_dir': current_dir,
         'ntask': 2
         }

policy = {
    'verbose': True,
    'type': PixelCostController,
    'replan_interval': 15,
    'num_samples': [600, 300],
    'selection_frac': 0.05,
    'predictor_propagation': True,   # use the model get the designated pixel for the next step!
    'initial_std': 0.005,
    'initial_std_lift': 0.05,  # std dev. in xy
    'initial_std_rot': np.pi / 10,
    'state_append': [0.41, 0.4, 0.184]
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
