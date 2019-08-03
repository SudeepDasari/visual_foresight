import numpy as np
import os
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.envs.robot_envs.autograsp_env import AutograspEnv
from visual_mpc.policy.cem_controllers.pixel_cost_controller import PixelCostController
from visual_mpc.envs.robot_envs.util.topic_utils import IMTopic
from visual_mpc.policy.cem_controllers.samplers import CorrelatedNoiseSampler


env_params = {
    'camera_topics': [IMTopic('/other/image_raw', flip=False)],                  #, IMTopic('/bot/image_raw'), IMTopic('/bot2/image_raw')],
    'cleanup_rate': -1,
    'save_video': True

}

agent = {'type' : BenchmarkAgent,
         'env': (AutograspEnv, env_params),
         'T': 15,  #number of commands per episodes (issued at control_rate / substeps HZ)
         'image_height': 48,
         'image_width': 64,
         'make_final_recording': ''
}

policy = {
    'type': PixelCostController,
    'replan_interval': 13,
    'zeros_for_start_frames': False,
    'start_planning': 2,
    'selection_frac': 2./3,
    'predictor_propagation': True,   # use the model get the designated pixel for the next step!
    'nactions': 13,
    "sampler": CorrelatedNoiseSampler,

    "model_params_path": "~/models/sawyer_single_view/experiment_state-2019-07-04_18-19-37.json",
    "model_restore_path": "~/models/sawyer_single_view/checkpoint_110000/model-110000",
}

config = {
    "experiment_name": "single_view_heldout",
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
