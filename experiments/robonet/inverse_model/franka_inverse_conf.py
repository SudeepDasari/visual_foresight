import numpy as np
import os
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.envs.robot_envs.autograsp_env import AutograspEnv
from visual_mpc.policy.cem_controllers.pixel_cost_controller import PixelCostController
from visual_mpc.envs.robot_envs.util.topic_utils import IMTopic
from visual_mpc.policy.cem_controllers.samplers import CorrelatedNoiseSampler
from visual_mpc.policy.inverse_models.inverse_model_base_controller import InvModelBaseController

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])


env_params = {
    #'email_login_creds': '.email_cred',
    'camera_topics': [IMTopic('/front/image_raw')],
    'robot_name':'franka',
    'robot_type':'franka',
    'gripper_attached':'hand',
    'cleanup_rate': -1,
    'duration': 3.5,
    'save_video': True
}

agent = {'type' : BenchmarkAgent,
         'env': (AutograspEnv, env_params),
         'T': 15,  #number of commands per episodes (issued at control_rate / substeps HZ)
         'image_height': 192,
         'image_width': 256,
         'make_final_recording': '',
         'goal_image_only':'',
         'no_goal_def':'',
	     'data_save_dir': BASE_DIR
}

policy = {
    'type': InvModelBaseController,
        "model_params_path": "/home/panda1/models/bigger_multibot",
        "model_restore_path":  "/home/panda1/models/bigger_multibot",


}

config = {
    "experiment_name": "inverse-model-onestep-replan-10",
    'traj_per_file':128,
    'save_data': True,
    'save_raw_images' : True,
    'start_index':0,
    'end_index': 30000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000,
    'nshuffle' : 200
}
