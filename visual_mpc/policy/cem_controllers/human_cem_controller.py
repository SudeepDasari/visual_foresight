from .cem_base_controller import CEMBaseController
from .visualizer.construct_html import save_gifs, save_img, save_html, fill_template
from visual_mpc.video_prediction.pred_util import get_context, rollout_predictions
import numpy as np
import imp
from collections import OrderedDict
import sys


if sys.version_info[0] == 2:
    input_fn = raw_input
else:
    input_fn = input


class HumanCEMController(CEMBaseController):
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        """

        :param ag_params: agent parameter dictionary
        :param policyparams: policy parameter dict
        :param gpu_id: gpu id
        :param ngpu: number of gpus to use
        """
        CEMBaseController.__init__(self, ag_params, policyparams)

        params = imp.load_source('params', ag_params['current_dir'] + '/conf.py')
        netconf = params.configuration
        self.predictor = netconf['setup_predictor'](ag_params, netconf, gpu_id, ngpu, self._logger)

        self._net_bsize = netconf['batch_size']
        self._net_context = netconf['context_frames']
        self._hp.start_planning = self._net_context
        self._n_cam = netconf['ncam']

        self._images, self._verbose_worker = None, None
        self._save_actions = None

    def reset(self):
        super(HumanCEMController, self).reset()
        self._save_actions = None
    
    def _default_hparams(self):
        default_dict = {
            "verbose_img_height": 128,
            'state_append': None,
        }
        parent_params = super(HumanCEMController, self)._default_hparams()

        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def evaluate_rollouts(self, actions, cem_itr):
        last_frames, last_states = get_context(self._net_context, self._t,
                                               self._state, self._images, self._hp)
        gen_images = rollout_predictions(self.predictor, self._net_bsize, actions,
                                                      last_frames, last_states, logger=self._logger)[0]
        gen_images = np.concatenate(gen_images, 0)

        verbose_folder = "planning_{}_itr_{}".format(self._t, cem_itr)
        content_dict = OrderedDict()
        # start images
        for c in range(self._n_cam):
            name = 'cam_{}_start'.format(c)
            save_path = save_img(self._verbose_worker, verbose_folder, name, self._images[-1, c])
            content_dict[name] = [save_path for _ in range(gen_images.shape[0])]

        # render predicted images
        for c in range(self._n_cam):
            verbose_images = [(gen_images[g_i, :, c] * 255).astype(np.uint8) for g_i in range(gen_images.shape[0])]
            row_name = 'cam_{}_pred_images'.format(c)
            content_dict[row_name] = save_gifs(self._verbose_worker, verbose_folder,
                                               row_name, verbose_images)

        html_page = fill_template(cem_itr, self._t, content_dict, img_height=self._hp.verbose_img_height)
        save_html(self._verbose_worker, "{}/preds.html".format(verbose_folder), html_page)

        scores = np.zeros(gen_images.shape[0])
        for i in range(gen_images.shape[0]):
            scores[i] = float(input_fn("Score for traj {}: ".format(i)))

        content_dict['scores'] = scores
        html_page = fill_template(cem_itr, self._t, content_dict, img_height=self._hp.verbose_img_height)
        save_html(self._verbose_worker, "{}/plan.html".format(verbose_folder), html_page)

        return scores

    def act(self, t=None, i_tr=None, images=None, state=None, verbose_worker=None):
        """
        Return a random action for a state.
        Args:
            if performing highres tracking images is highres image
            t: the current controller's Time step
            goal_pix: in coordinates of small image
            desig_pix: in coordinates of small image
        """
        if t <= 0 and 'y' == input_fn("restore traj?: "):
            import cPickle as pkl
            self._save_actions = pkl.load(open(input_fn('path:'), 'rb'))
            import pdb; pdb.set_trace()
        
        if self._save_actions is not None and t < len(self._save_actions):
            return {'actions': self._save_actions[t]['actions']}

        self._images = images
        self._verbose_worker = verbose_worker

        return super(HumanCEMController, self).act(t, i_tr, state)
