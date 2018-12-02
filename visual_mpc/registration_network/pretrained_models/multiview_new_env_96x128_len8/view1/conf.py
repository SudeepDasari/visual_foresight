import python_visual_mpc
base_dir = python_visual_mpc.__file__

base_dir = '/'.join(str.split(base_dir, '/')[:-2])

# tf record data location:
import os
DATA_DIR = {
         os.environ['VMPC_DATA_DIR'] + '/sawyer_grasping/ag_long_records_15kfullres/good': 32,
         os.environ['VMPC_DATA_DIR'] + '/sawyer_grasping/ag_long_records_15kfullres/bad': 32,
        }
import os
current_dir = os.path.dirname(os.path.realpath(__file__))

# local output directory
OUT_DIR = current_dir + '/modeldata'


configuration = {
'experiment_name': 'correction',
'data_dir': DATA_DIR,       # 'directory containing data.' ,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': base_dir,   #'directory for writing summary.' ,
'num_iterations':100000,
'sequence_length':30,
'max_deltat':8,
'train_val_split':.95,
'visualize':'',
'skip_frame':1,
'batch_size': 64,           #'batch size for training' ,
'learning_rate': 0.001,     #'the base learning rate of the generator' ,
'normalization':'None',
'image_only':'',
'orig_size': [96,128],
'norm':'charbonnier',
'smoothcost':1e-7,
'smoothmode':'2nd',
'fwd_bwd':'',
'flow_diff_cost':1e-7,
'hard_occ_thresh':'',
'occlusion_handling':1e-7,
'occ_thres_mult':0.5,
'occ_thres_offset':1.,
'flow_penal':1e-7,
'ch_mult':4,
'decay_lr':'',
'view':1,
'new_loader': True
}