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
         'T': 14,  #number of commands per episodes (issued at control_rate / substeps HZ)
         'image_height': 48,
         'image_width': 64,
         'make_final_recording': ''
}

policy = {
    'type': PixelCostController,
    'replan_interval': 13,
    'verbose_every_iter': True,
    'zeros_for_start_frames': False,
    'num_samples': 600,
    'selection_frac': 2./3,
    'predictor_propagation': True,   # use the model get the designated pixel for the next step!
    'nactions': 13,
    "sampler": CorrelatedNoiseSampler,
    'context_action_weight': [2, 2, 0.05, 2],
    'initial_std': [0.05, 0.05, 0.2, np.pi / 10, 1],

    "model_params_path": "~/models/sawyer_only/experiment_state-2019-06-26_01-33-31.json",
    "model_restore_path": "~/models/sawyer_only/checkpoint_210000/model-210000",
}

config = {
    "experiment_name": "all_views_heldout",
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
