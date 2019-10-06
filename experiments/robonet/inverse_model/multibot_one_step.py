import numpy as np
import os
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.envs.robot_envs.autograsp_env import AutograspEnv
from visual_mpc.policy.cem_controllers.pixel_cost_controller import PixelCostController
from visual_mpc.envs.robot_envs.util.topic_utils import IMTopic
from visual_mpc.policy.cem_controllers.samplers import CorrelatedNoiseSampler
from visual_mpc.policy.inverse_models.inverse_model_base_controller import InvModelBaseController

env_params = {
    'camera_topics': [IMTopic('/front/image_raw', flip=False)],                  #, IMTopic('/bot/image_raw'), IMTopic('/bot2/image_raw')],
    'cleanup_rate': -1,
    'save_video': True,
    'gripper_attached': 'none'
}

agent = {'type' : BenchmarkAgent,
         'env': (AutograspEnv, env_params),
         'T': 10, #14,  #number of commands per episodes (issued at control_rate / substeps HZ)
         'image_height': 192,
         'image_width': 256,
         'make_final_recording': '',
         'goal_image_only':'',
         #'load_goal_image':'/home/sudeep/goal_images/side_noinsert.jpg',
         'no_goal_def':'',
	     'data_save_dir': 'outputs/'
}

policy = {
    'type': InvModelBaseController,
    "model_params_path":"~/models/multibot_disc_bigger_nf/InverseTrainable_0_fe2f43ee_2019-10-04_08-52-392ofutdfh/checkpoint_140000",
    "model_restore_path": "~/models/multibot_disc_bigger_nf/InverseTrainable_0_fe2f43ee_2019-10-04_08-52-392ofutdfh/checkpoint_140000",
    "replan_every": 1,
}

config = {
    "experiment_name": "inverse-model-multibot",
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
