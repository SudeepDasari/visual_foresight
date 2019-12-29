import os.path
from visual_mpc.policy.cem_controllers import PixelCostController
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.envs.mujoco_env.cartgripper_env.cartgripper_xz_grasp import CartgripperXZGrasp

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))


env_params = {
    # resolution sufficient for 16x anti-aliasing
    'viewer_image_height': 96,
    'viewer_image_width': 128,
    'cube_objects': True
}


agent = {
    'type': BenchmarkAgent,
    'env': (CartgripperXZGrasp, env_params),
    'data_save_dir': BASE_DIR,
    'T': 45,
    'image_height': 48,
    'image_width': 64,
    'make_final_gif_pointoverlay': True,
    'record':  BASE_DIR + '/record/',
    'start_goal_confs': os.environ['VMPC_DATA_DIR'] + '/cartgripper_xz_grasp/lifting_tasks',
    'current_dir': current_dir
}

policy = {
    'type': PixelCostController,
    'action_order': ['x', 'z', 'grasp'],
    'initial_std_lift': 0.5,  # std dev. in xy
    'rejection_sampling': False,
    'replan_interval': 10,
    'num_samples': 800

}

config = {
    'traj_per_file': 128,
    'current_dir': current_dir,
    'save_data': True,
    'seperate_good': False,
    'save_raw_images': True,
    'start_index': 0,
    'end_index': 5,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}
