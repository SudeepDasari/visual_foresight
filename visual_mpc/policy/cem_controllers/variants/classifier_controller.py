from visual_mpc.policy.cem_controllers import CEMBaseController
import imp
import control_embedding
import numpy as np
from visual_mpc.video_prediction.pred_util import get_context, rollout_predictions
from ..visualizer.construct_html import save_gifs, save_html, save_img, fill_template, img_entry_html
from collections import OrderedDict


LOG_SHIFT = 1e-5

class ClassifierController(CEMBaseController):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        """

        :param ag_params: agent parameters
        :param policyparams: policy parameters
        :param gpu_id: starting gpu id
        :param ngpu: number of gpus
        """
        CEMBaseController.__init__(self, ag_params, policyparams)

        params = imp.load_source('params', ag_params['current_dir'] + '/conf.py')
        net_conf = params.configuration

        if ngpu > 1:
            vpred_ngpu = ngpu - 1
        else: vpred_ngpu = ngpu

        self._predictor = net_conf['setup_predictor'](ag_params, net_conf, gpu_id, vpred_ngpu, self._logger)
        self._scoring_func = control_embedding.deploy_simple_model(self._hp.classifier_conf_path, batch_size=self._hp.classifier_batch_size,
                                                                   restore_path=self._hp.classifier_restore_path,
                                                                   device_id=gpu_id + ngpu - 1)

        self._vpred_bsize = net_conf['batch_size']

        self._seqlen = net_conf['sequence_length']
        self._net_context = net_conf['context_frames']
        self._hp.start_planning = self._net_context            # skip steps so there are enough context frames
        self._n_pred = self._seqlen - self._net_context
        assert self._n_pred > 0, "context_frames must be larger than sequence_length"

        self._img_height, self._img_width = net_conf['orig_size']

        self._n_cam = net_conf['ncam']

        self._images = None
        self._expert_images = None
        self._expert_score = None
        self._goal_image = None
        self._start_image = None
        self._verbose_worker = None

    def reset(self):
        self._expert_score = None
        self._images = None
        self._expert_images = None
        self._goal_image = None
        self._start_image = None
        self._verbose_worker = None
        return super(ClassifierController, self).reset()

    def _default_hparams(self):
        default_dict = {
            'finalweight': 100,
            'classifier_conf_path': '',
            'classifier_restore_path': '',
            'classifier_batch_size': 200,
            'state_append': None,
            'compare_to_expert': False,
            'verbose_img_height': 128,
            'verbose_frac_display': 0.
        }
        parent_params = super(ClassifierController, self)._default_hparams()

        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def evaluate_rollouts(self, actions, cem_itr):
        previous_actions = np.concatenate([x[None] for x in self._sampler.chosen_actions[-self._net_context:]], axis=0)
        previous_actions = np.tile(previous_actions, [actions.shape[0], 1, 1])
        input_actions = np.concatenate((previous_actions, actions), axis=1)[:, :self._seqlen]

        last_frames, last_states = get_context(self._net_context, self._t,
                                               self._state, self._images, self._hp)

        gen_images = rollout_predictions(self._predictor, self._vpred_bsize, input_actions,
                                         last_frames, last_states, logger=self._logger)[0]

        gen_images = np.concatenate(gen_images, 0) * 255.

        raw_scores = np.zeros((self._n_cam, actions.shape[0], self._n_pred))
        for c in range(self._n_cam):
            input_images = gen_images[:, :, c].reshape((-1, self._img_height, self._img_width, 3)) / 255
            logits = self._scoring_func(input_images)['logits']
            logits = logits.reshape((actions.shape[0], self._n_pred, 2))
            import pdb; pdb.set_trace()
            raw_scores[c] = -np.log(logits[:, :, 1] + LOG_SHIFT)

        raw_scores = np.sum(raw_scores, axis=0)
        scores = self._weight_scores(raw_scores)

        if self._verbose_condition(cem_itr):
            verbose_folder = "planning_{}_itr_{}".format(self._t, cem_itr)
            content_dict = OrderedDict()
            visualize_indices = scores.argsort()[:10]

            # start images
            for c in range(self._n_cam):
                name = 'cam_{}_start'.format(c)
                save_path = save_img(self._verbose_worker, verbose_folder, name, self._images[-1, c])
                content_dict[name] = [save_path for _ in visualize_indices]

            # render predicted images
            for c in range(self._n_cam):
                verbose_images = [(gen_images[g_i, :, c]).astype(np.uint8) for g_i in visualize_indices]
                row_name = 'cam_{}_pred_images'.format(c)
                content_dict[row_name] = save_gifs(self._verbose_worker, verbose_folder,
                                                       row_name, verbose_images)

            # save scores
            content_dict['scores'] = scores[visualize_indices]
            html_page = fill_template(cem_itr, self._t, content_dict, img_height=self._hp.verbose_img_height)
            save_html(self._verbose_worker, "{}/plan.html".format(verbose_folder), html_page)

            html_page = fill_template(cem_itr, self._t, content_dict, img_height=self._hp.verbose_img_height)
            save_html(self._verbose_worker, "{}/plan.html".format(verbose_folder), html_page)

        return scores

    def _weight_scores(self, raw_scores):
        if self._hp.finalweight >= 0:
            scores = raw_scores.copy()
            scores[:, -1] *= self._hp.finalweight
            scores = np.sum(scores, axis=1) / sum([1. for _ in range(self._n_pred - 1)] + [self._hp.finalweight])
        else:
            scores = raw_scores[:, -1].copy()
        return scores


    def act(self, t=None, i_tr=None, images=None, state=None, verbose_worker=None):
        self._images = images
        self._verbose_worker = verbose_worker

        return super(ClassifierController, self).act(t, i_tr, state)
