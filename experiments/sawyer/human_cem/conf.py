from visual_mpc.video_prediction.setup_predictor import setup_predictor
from visual_mpc.video_prediction.vpred_model_interface import VPred_Model_Interface
from video_prediction.models.indep_multi_savp_model import IndepMultiSAVPVideoPredictionModel
import video_prediction

base_dir = video_prediction.__file__
base_dir = '/'.join(str.split(base_dir, '/')[:-2])
modeldir = base_dir + '/pretrained_models/mixed_datasets/towel_hard_objects/'

configuration = {
'pred_model': VPred_Model_Interface,
'pred_model_class':IndepMultiSAVPVideoPredictionModel,
'setup_predictor':setup_predictor,
'json_dir':  modeldir + '/view0/model.savp.None',
'pretrained_model':[modeldir + '/view0/model.savp.None/model-300000', modeldir + '/view1/model.savp.None/model-300000'],   # 'filepath of a pretrained model to resume training from.' ,
'sequence_length': 15,      # 'sequence length to load, including context frames.' ,
'context_frames': 2,        # of frames before predictions.' ,
'model': 'appflow',            #'model architecture to use - CDNA, DNA, or STP' ,
'batch_size': 200,
'sdim':8,
'adim':4,
'orig_size':[48,64],
'ndesig':1,
'ncam':2,
}
