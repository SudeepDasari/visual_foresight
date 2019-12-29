import numpy as np
import os
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.envs.robot_envs.autograsp_env import AutograspEnv
from visual_mpc.policy.cem_controllers.pixel_cost_controller import PixelCostController
from visual_mpc.envs.robot_envs.util.topic_utils import IMTopic
from visual_mpc.policy.cem_controllers.samplers import CorrelatedNoiseSampler


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])

env_params = {
    #'email_login_creds': '.email_cred',
    'camera_topics': [IMTopic('/front/image_raw')],
                      # IMTopic('/left/image_raw')],
                      # IMTopic('/right_side/image_raw')],
                      # IMTopic('/left_side/image_raw')],
                      # IMTopic('/right/image_raw')],
    'robot_name':'franka',
    'robot_type':'franka',
    'gripper_attached':'hand',
    'cleanup_rate': -1,
    'duration': 3.5,
    'reopen':False,
    'save_video': True
}

agent = {'type' : BenchmarkAgent,
         'env': (AutograspEnv, env_params),
         'data_save_dir': BASE_DIR,
         'T': 15,  #number of commands per episodes (issued at control_rate / substeps HZ)
         'image_height': 48,
         'image_width': 64,
         'make_final_recording': ''
}

policy = {
    'type': PixelCostController,
    'verbose_every_iter': True,
    'zeros_for_start_frames':False,
    'replan_interval': 10,
    # 'num_samples': 200,
    'start_planning': 5,
    'iterations':5,
    'selection_frac': 1./10,
    # 'predictor_propagation': True,   # use the model get the designated pixel for the next step!
    'nactions': 10,

    # "model_path": "~/models/franka_sanity/sanity_check_model/checkpoint_415000/", # 8K
     # "model_path": "/home/panda1/models/checkpoint_145000/", # 400
          "model_path": "/home/panda1/models/ag_franka/VPredTrainable_0_462f7842_2019-10-05_00-03-46poiv7dyy/checkpoint_75000/", # Fineture


    "sampler": CorrelatedNoiseSampler
}

config = {
    'experiment_name': 'franka_sanity_check',
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
