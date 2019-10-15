from visual_mpc.video_prediction.setup_predictor import setup_predictor
from visual_mpc.video_prediction.vpred_model_interface import VPred_Model_Interface
from video_prediction.models.savp_model import SAVPVideoPredictionModel
import robonet


modeldir = '/home/sudeep/Documents/video_prediction/pretrained_models/mixed_datasets/towel_hard_objects/view0/'

configuration = {
'pred_model': VPred_Model_Interface,
'pred_model_class': SAVPVideoPredictionModel,
'setup_predictor':setup_predictor,
'json_dir':  modeldir + '/model.savp.None',
'pretrained_model':modeldir + '/model.savp.None/model-300000',   # 'filepath of a pretrained model to resume training from.' ,
'sequence_length': 15,      # 'sequence length to load, including context frames.' ,
'context_frames': 2,        # of frames before predictions.' ,
'model': 'appflow',            #'model architecture to use - CDNA, DNA, or STP' ,
'batch_size': 50,
'sdim':8,
'adim':4,
'orig_size':[48,64],
'no_pix_distrib': '',
'ncam': 1
}
