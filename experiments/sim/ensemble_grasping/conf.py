from visual_mpc.video_prediction.setup_predictor import setup_predictor
from visual_mpc.video_prediction.vpred_model_interface import VPred_Model_Interface
from video_prediction.models.indep_ensemble_savp_model import IndepEnsembleSAVPVideoPredictionModel
import video_prediction


base_dir = video_prediction.__file__
base_dir = '/'.join(str.split(base_dir, '/')[:-2])
modeldir = base_dir + '/pretrained_models/ensemble_tests/train_cartgripper'

configuration = {
'pred_model': VPred_Model_Interface,
'pred_model_class':IndepEnsembleSAVPVideoPredictionModel,
'setup_predictor':setup_predictor,
'json_dir':  modeldir + '/view0/model.ensemble_savp.None/',
'pretrained_model':[modeldir + '/view0/model.ensemble_savp.None/model-280000', modeldir + '/view1/model.ensemble_savp.None/model-275000'],   # 'filepath of a pretrained model to resume training from.' ,
'sequence_length': 15,      # 'sequence length to load, including context frames.' ,
'context_frames': 2,        # of frames before predictions.' ,
'model': 'appflow',            #'model architecture to use - CDNA, DNA, or STP' ,
'batch_size': 400,
'sdim':5,
'adim':4,
'orig_size':[48,64],
'ndesig':1,
'ncam':2,
}
