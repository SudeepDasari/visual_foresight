import numpy as np
import os
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.envs.robot_envs.autograsp_env import AutograspEnv
from visual_mpc.policy.cem_controllers.pixel_cost_controller import PixelCostController
from visual_mpc.envs.robot_envs.util.topic_utils import IMTopic
from visual_mpc.policy.cem_controllers.samplers import CorrelatedNoiseSampler
from visual_mpc.policy.inverse_models.inverse_model_base_controller import InvModelBaseController

env_params = {
    'camera_topics': [IMTopic('/front/image_raw', flip=True)],                  #, IMTopic('/bot/image_raw'), IMTopic('/bot2/image_raw')],
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
    #"model_params_path": "/home/stephen/models/InverseTrainable_0_534e53f2_2019-09-22_02-07-585pwzvrv_/checkpoint_130000",

    "model_params_path": "/home/stephen/models/onestep_context/checkpoint_170000",
    # "model_restore_path": "~/models/inverse_model/sudeep_training/InverseTrainable_2_load_T=7_4f3b737c_2019-08-18_08-15-55u3mucwsl/checkpoint_95000/model-95000",
    "model_restore_path": "/home/stephen/models/onestep_context/checkpoint_170000"
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
