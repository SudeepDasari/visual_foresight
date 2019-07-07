import numpy as np
import os
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.envs.robot_envs.vanilla_env import VanillaEnv
from visual_mpc.policy.cem_controllers.pixel_cost_controller import PixelCostController
from visual_mpc.envs.robot_envs.util.topic_utils import IMTopic
from visual_mpc.policy.cem_controllers.samplers import AutograspSampler


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))


env_params = {
    'camera_topics': [IMTopic('/front/image_raw')],
    'robot_type': 'baxter',
    'gripper_attached': 'baxter_gripper',
    'cleanup_rate': -1,
    'duration': 2.5
}

agent = {'type' : BenchmarkAgent,
         'env': (VanillaEnv, env_params),
         'data_save_dir': BASE_DIR,
         'T': 14,  #number of commands per episodes (issued at control_rate / substeps HZ)
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
    'repeat': 1,

    "model_params_path": "~/models/run_lr1e-4/experiment_state-2019-07-04_21-59-09.json",
    "model_restore_path": "~/models/run_lr1e-4/checkpoint_80000/model-80000",

    "sampler": AutograspSampler,
    'action_norm_factor': 714.285714286,
    'gripper_close_cmd': 100,
    'gripper_open_cmd': 0

}

config = {
    'experiment_name': 'baxter_scratch',
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
