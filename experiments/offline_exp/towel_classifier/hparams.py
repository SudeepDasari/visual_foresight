import numpy as np
import os
from visual_mpc.envs.offline_env import OfflineSawyerEnv
from visual_mpc.agent.offline_agent import OfflineAgent
from visual_mpc.policy.cem_controllers.samplers.folding_sampler import FoldingCEMSampler
from visual_mpc.policy.cem_controllers.variants.classifier_controller import ClassifierController
BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))


env_params = {
}

agent = {'type' : OfflineAgent,
         'env': (OfflineSawyerEnv, env_params),
         'data_save_dir': BASE_DIR,
         'T' : 15,  #number of commands per episodes (issued at control_rate / substeps HZ)
         'image_height': 48,
         'image_width': 64,
         'current_dir': current_dir,
         'no_goal_def': True
         }

policy = {
    'type': ClassifierController,
    'replan_interval': 15,
    'num_samples': 600,
    'selection_frac': 0.05,
    'sampler': FoldingCEMSampler,
    # 'predictor_propagation': True,   # use the model get the designated pixel for the next step!
    'initial_std': 0.005,
    'initial_std_lift': 0.05,  # std dev. in xy
    'classifier_conf_path': '/home/sudeep/Documents/control_embedding/experiments/towel_exp/base.json',
    'classifier_restore_path': '/home/sudeep/Documents/control_embedding/experiments/towel_exp/base_model/model-10000',
    'classifier_batch_size': 50,
    'verbose_every_iter': True,
    'state_append': [0.41, 0.25, 0.166]
    
}

config = {
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
