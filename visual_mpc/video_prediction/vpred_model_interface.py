import tensorflow as tf
from video_prediction.models import MultiSAVPVideoPredictionModel, SAVPVideoPredictionModel, IndepMultiSAVPVideoPredictionModel
import json
import os


class VPred_Model_Interface:
    def __init__(self,
                 conf = None,
                 images=None,
                 actions=None,
                 states=None,
                 pix_distrib= None,
                 mode='test',
                 build_loss = False,
                 load_data = True
                 ):

        self.iter_num = tf.placeholder(tf.float32, [])
        with open(os.path.join(conf['json_dir'], "model_hparams.json")) as f:
            model_hparams_dict = json.loads(f.read())
            model_hparams_dict.pop('num_gpus', None)  # backwards-compatibility
            if 'override_json' in conf:
                model_hparams_dict.update(conf['override_json'])
        with open(os.path.join(conf['json_dir'], "dataset_hparams.json")) as f:
            datatset_hparams_dict = json.loads(f.read())

        if 'autograsp' in datatset_hparams_dict:
            self.adim = datatset_hparams_dict['autograsp']
        else: self.adim = conf['adim']

        self.sdim = conf['sdim']
        seq_len = model_hparams_dict['sequence_length']
        nctxt = model_hparams_dict['context_frames']
        ncam = conf['ncam']
        self.img_height, self.img_width = conf['orig_size']
        if images is None:
            self.actions_pl = tf.placeholder(tf.float32, name='actions',
                                             shape=(conf['batch_size'], seq_len, self.adim))
            actions = self.actions_pl
            self.states_pl = tf.placeholder(tf.float32, name='states',
                                            shape=(conf['batch_size'], seq_len, self.sdim))
            states = self.states_pl
            self.images_pl = tf.placeholder(tf.float32, name='images',
                                            shape=(conf['batch_size'], seq_len, ncam, self.img_height, self.img_width, 3))
            images = self.images_pl

            targets = images[:,nctxt:, 0]  #this is a hack to make Alexmodel compute losses. It will take take second image automatically.
        else:
            targets = None

        if 'pred_model_class' in conf:
            self.m = conf['pred_model_class'](mode=mode, hparams_dict=model_hparams_dict, num_gpus=1)
        else:
            if ncam == 1:
                self.m = SAVPVideoPredictionModel(mode=mode, hparams_dict=model_hparams_dict, num_gpus=1)
            elif ncam == 2:
                self.m = MultiSAVPVideoPredictionModel(mode=mode, hparams_dict=model_hparams_dict, num_gpus=1)

        inputs = {'actions':actions ,'images':images[:,:,0]}
        use_state = datatset_hparams_dict.get('use_state', False)
        if use_state:
            inputs['states'] = states

        if ncam == 2:
            inputs['images1'] = images[:,:,1]

        if pix_distrib is not None: # input batch , t, ncam, r, c, ndesig
            inputs['pix_distribs'] = pix_distrib[:,:,0]
            if ncam == 2:
                inputs['pix_distribs1'] = pix_distrib[:,:,1]

        self.m.build_graph(inputs, targets)

        gen_images = [self.m.outputs['gen_images']]
        if ncam == 2:
            gen_images.append(self.m.outputs['gen_images1'])
        self.gen_images = tf.stack(gen_images, axis=2) #ouput  b, t, ncam, r, c, 3

        if use_state:
            self.gen_states = self.m.outputs['gen_states']
        else: self.gen_states = None

        if pix_distrib is not None:
            gen_distrib = [self.m.outputs['gen_pix_distribs']]
            if ncam == 2:
                gen_distrib.append(self.m.outputs['gen_pix_distribs1'])
            self.gen_distrib = tf.stack(gen_distrib, axis=2) #ouput  b, t, ncam, r, c, 3
