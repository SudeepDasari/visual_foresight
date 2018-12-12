from visual_mpc.sim.util.config_agent import CreateConfigAgent
from visual_mpc.envs.mujoco_env.cartgripper_env.autograsp_env import AutograspCartgripperEnv
from visual_mpc.policy.policy import DummyPolicy
import os


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))


env_params = {
    'num_objects': 1,
    'object_mass': 0.5,
    'friction': 1.0,
    'finger_sensors': True,
    'minlen': 0.03,
    'maxlen': 0.045,
    'object_object_mindist': 0.15,
    'cube_objects': True,
    'autograsp': {'zthresh': -0.06, 'touchthresh': 0.0, 'reopen': True}
}


agent = {
    'type': CreateConfigAgent,
    'env': (AutograspCartgripperEnv, env_params),
    'data_save_dir': BASE_DIR,
    'image_height': 48,
    'T': 1,
    'image_width': 64,
    'gen_xml': 1,  # generate xml every nth trajecotry
}


config = {
    'current_dir': current_dir,
    'agent': agent,
    'policy': {'type': DummyPolicy},
    'save_raw_images': True,
    'start_index': 0,
    'end_index': 100,
    'ngroup': 1000
}