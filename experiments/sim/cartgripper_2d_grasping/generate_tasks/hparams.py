""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.policy.handcrafted.lifting_policy import LiftingPolicy
from visual_mpc.agent.general_agent import GeneralAgent
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
    'type': GeneralAgent,
    'env': (CartgripperXZGrasp, env_params),
    'data_save_dir': BASE_DIR,
    'T': 15,
    'image_height' : 48,
    'image_width' : 64,
    'gen_xml': 1,   #generate xml every nth trajecotry
    'make_final_gif': '',
    'rejection_sample': 500,
    'save_reset_data': True
}

policy = {
    'type' : LiftingPolicy,
    'sigma': [0.0, 0., 0],
    'frac_act': [0.4, 0.2],
    'bounds': [[-0.4, 0.1], [0.4, 0.15]],
}

config = {
    'current_dir' : current_dir,
    'save_data': True,
    'seperate_good': False,
    'save_raw_images' : True,
    'start_index':0,
    'end_index': 10,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}
