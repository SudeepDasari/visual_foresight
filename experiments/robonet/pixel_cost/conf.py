from visual_mpc.video_prediction.setup_predictor import setup_predictor
from visual_mpc.video_prediction.vpred_model_interface import VPred_Model_Interface
from video_prediction.models.savp_model import SAVPVideoPredictionModel
import video_prediction

base_dir = video_prediction.__file__
base_dir = '/'.join(str.split(base_dir, '/')[:-2])
modeldir = base_dir + '/robonet_experiments/sawyer/short_context'

configuration = {
'pred_model': VPred_Model_Interface,
'pred_model_class': SAVPVideoPredictionModel,
'setup_predictor':setup_predictor,
'json_dir':  modeldir + '/model.savp.None',
'pretrained_model':modeldir + '/model.savp.None/model-190000',   # 'filepath of a pretrained model to resume training from.' ,
'sequence_length': 13,      # 'sequence length to load, including context frames.' ,
'context_frames': 2,        # of frames before predictions.' ,
'model': 'appflow',            #'model architecture to use - CDNA, DNA, or STP' ,
'batch_size': 200,
'sdim':5,
'adim':4,
'orig_size':[48,64],
'ndesig':1,
'ncam':1,
}
