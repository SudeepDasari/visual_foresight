from visual_mpc.policy.cem_controllers import CEMBaseController
import imp
import control_embedding
import numpy as np
from visual_mpc.video_prediction.pred_util import get_context, rollout_predictions
from ..visualizer.construct_html import save_gifs, save_html, save_img, fill_template, img_entry_html
from ..visualizer.plot_helper import plot_score_hist
from collections import OrderedDict


class NCECostController(CEMBaseController):
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
        self._scoring_func = control_embedding.deploy_model(self._hp.nce_conf_path, batch_size=self._hp.nce_batch_size,
                                                            restore_path=self._hp.nce_restore_path,
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
        return super(NCECostController, self).reset()

    def _default_hparams(self):
        default_dict = {
            'score_fn': 'dot_prod',
            'finalweight': 100,
            'nce_conf_path': '',
            'nce_restore_path': '',
            'nce_batch_size': 200,
            'state_append': None,
            'compare_to_expert': False,
            'verbose_img_height': 128,
            'verbose_frac_display': 0.
        }
        parent_params = super(NCECostController, self)._default_hparams()

        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def evaluate_rollouts(self, actions, cem_itr):
        last_frames, last_states = get_context(self._net_context, self._t,
                                               self._state, self._images, self._hp)

        gen_images = rollout_predictions(self._predictor, self._vpred_bsize, actions,
                                         last_frames, last_states, logger=self._logger)[0]

        gen_images = np.concatenate(gen_images, 0) * 255.

        raw_scores = np.zeros((self._n_cam, actions.shape[0], self._n_pred))
        for c in range(self._n_cam):
            goal, start = self._goal_image[c][None], self._start_image[c][None]
            input_images = gen_images[:, :, c].reshape((-1, self._img_height, self._img_width, 3))
            embed_dict = self._scoring_func(goal, start, input_images)

            gs_enc = embed_dict['goal_enc'][0][None]
            in_enc = embed_dict['input_enc'].reshape((actions.shape[0], self._n_pred, -1))
            raw_scores[c] = self._eval_embedding_cost(gs_enc, in_enc)

        raw_scores = np.sum(raw_scores, axis=0)
        scores = self._weight_scores(raw_scores)

        if self._verbose_condition(cem_itr):
            verbose_folder = "planning_{}_itr_{}".format(self._t, cem_itr)
            content_dict = OrderedDict()
            visualize_indices = scores.argsort()[:max(10, int(actions.shape[0] * self._hp.verbose_frac_display))]

            # start image and predictions (alternate by camera)
            for c in range(self._n_cam):
                name = 'cam_{}_start'.format(c)
                save_path = save_img(self._verbose_worker, verbose_folder, name, self._images[-1, c])
                content_dict[name] = [save_path for _ in visualize_indices]

                name = 'cam_{}_goal'.format(c)
                save_path = save_img(self._verbose_worker, verbose_folder, name, self._goal_image[c].astype(np.uint8))
                content_dict[name] = [save_path for _ in visualize_indices]

                verbose_images = [gen_images[g_i, :, c].astype(np.uint8) for g_i in visualize_indices]
                row_name = 'cam_{}_pred_images'.format(c)
                content_dict[row_name] = save_gifs(self._verbose_worker, verbose_folder,
                                                   row_name, verbose_images)

            # scores
            content_dict['scores'] = scores[visualize_indices]
            content_dict['NCE Res'] = raw_scores[visualize_indices]

            if self._hp.compare_to_expert and self._expert_score is None:
                expert_scores = np.zeros((self._n_cam, 1, self._n_pred))
                for c in range(self._n_cam):
                    expert_goal, expert_start = self._expert_images[-1][c], self._expert_images[0][c]
                    embed_dict = self._scoring_func(expert_goal[None], expert_start[None], self._expert_images[:,c])

                    gs_enc = embed_dict['goal_enc'][0][None]
                    in_enc = embed_dict['input_enc'].reshape((1, self._n_pred, -1))
                    expert_scores[c] = self._eval_embedding_cost(gs_enc, in_enc)

                self._expert_score = self._weight_scores(np.sum(expert_scores, axis=0))[0]

            hist = plot_score_hist(scores, tick_value=self._expert_score)
            hist_path = save_img(self._verbose_worker, verbose_folder, "score_histogram", hist)
            extra_entry = img_entry_html(hist_path, height=hist.shape[0], caption="score histogram")

            html_page = fill_template(cem_itr, self._t, content_dict, img_height=self._hp.verbose_img_height,
                                      extra_html=extra_entry)
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

    def _eval_embedding_cost(self, goal_embed, input_embed):
        if self._hp.score_fn == 'dot_prod':
            # - log prob ignoring constant term (denominator)
            return -np.matmul(goal_embed[None], np.swapaxes(input_embed, 2, 1))[:, 0]
        raise NotImplementedError

    def act(self, t=None, i_tr=None, goal_image=None, images=None, state=None, verbose_worker=None):
        self._start_image = images[-1].astype(np.float32)
        self._goal_image = goal_image[-1] * 255
        self._images = images
        self._verbose_worker = verbose_worker

        if self._hp.compare_to_expert:
            self._expert_images = goal_image[1:self._n_pred + 1] * 255

        return super(NCECostController, self).act(t, i_tr, state)
