import numpy as np
import os
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.envs.robot_envs.autograsp_env import AutograspEnv
from visual_mpc.policy.cem_controllers.pixel_cost_controller import PixelCostController
from visual_mpc.envs.robot_envs.util.topic_utils import IMTopic


env_params = {
    'camera_topics': [IMTopic('/front/image_raw', flip=True)],                  #, IMTopic('/bot/image_raw'), IMTopic('/bot2/image_raw')],
    'gripper_attached': 'none',
    'cleanup_rate': -1,
    'duration': 2.5
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
    'num_samples': 600,
    'selection_frac': 0.05,
    'predictor_propagation': True,   # use the model get the designated pixel for the next step!
    'initial_std_lift': 0.2,  # std dev. in xy
    'initial_std_rot': np.pi / 10,
    'rejection_sampling': False,
    'nactions': 13,
    'repeat': 1,

    "model_params_path": "~/robotiq_models/robotiq_startall/experiment_state-2019-07-07_00-00-26.json",
    "model_restore_path": "~/robotiq_models/robotiq_startall/start_all/checkpoint_70000/model-190000",

}

config = {
    "experiment_name": "fine_tune_start_all",
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
