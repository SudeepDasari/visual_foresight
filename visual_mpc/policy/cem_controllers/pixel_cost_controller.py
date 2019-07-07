import numpy as np
import imp
import os
import cv2
from .cem_base_controller import CEMBaseController
from .visualizer.construct_html import save_gifs, save_html, save_img, fill_template
import matplotlib.pyplot as plt
from collections import OrderedDict
from visual_mpc.video_prediction.pred_util import get_context, rollout_predictions
try:
    from robonet.video_prediction.testing import VPredEvaluation
    DefaultPredClass = VPredEvaluation
except ImportError:
    DefaultPredClass = None

class PixelCostController(CEMBaseController):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        """

        :param ag_params: agent parameter dictionary
        :param policyparams: policy parameter dict
        :param gpu_id: gpu id
        :param ngpu: number of gpus to use
        """
        CEMBaseController.__init__(self, ag_params, policyparams)
        predictor_hparams = {}
        predictor_hparams['adim'] = self._adim
        predictor_hparams['sdim'] = self._sdim
        predictor_hparams['designated_pixel_count'] = self._hp.designated_pixel_count
        predictor_hparams['run_batch_size'] = min(self._hp.vpred_batch_size, self._hp.num_samples)
        predictor_hparams['img_dims'] = [ag_params['image_height'], ag_params['image_width']]

        self.predictor = self._hp.predictor_class(self._hp.model_params_path, predictor_hparams, n_gpus=ngpu, first_gpu=gpu_id)
        self.predictor.restore(self._hp.model_restore_path)

        self._net_context = self.predictor.n_context
        if self._hp.start_planning < self._net_context - 1:
            self._hp.start_planning = self._net_context - 1

        self._n_desig = self._hp.designated_pixel_count
        self._img_height, self._img_width = [ag_params['image_height'], ag_params['image_width']]

        self._n_cam = 1 #self.predictor.n_cam

        self._desig_pix = None
        self._goal_pix = None
        self._images = None

        if self._hp.predictor_propagation:
            self._chosen_distrib = None  # record the input distributions

    def _default_hparams(self):
        default_dict = {
            'predictor_class': DefaultPredClass,
            'model_params_path': '',
            'model_restore_path': '',
            'vpred_batch_size': 200,
            'designated_pixel_count': 1,

            "verbose_img_height": 128,
            'predictor_propagation':False,
            'only_take_first_view':False,
            'state_append': None,
            'finalweight': 10.
        }
        parent_params = super(PixelCostController, self)._default_hparams()

        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def reset(self):
        super(PixelCostController, self).reset()
        if self._hp.predictor_propagation:
            self._chosen_distrib = None  # record the input distributions

    def evaluate_rollouts(self, actions, cem_itr):
        context = {
            "context_frames": self._images,
            "context_actions": self._sampler.chosen_actions,
            "context_pixel_distributions": self._make_input_distrib(cem_itr),
            "context_states": self._state
        }
        prediction_dict = self.predictor(context, {'actions': actions})
        gen_images, gen_distrib = [prediction_dict[k] for k in  ['predicted_frames', 'predicted_pixel_distributions']]

        scores = self._eval_pixel_cost(cem_itr, gen_distrib, gen_images)
        
        if self._verbose_condition(cem_itr):
            verbose_folder = "planning_{}_itr_{}".format(self._t, cem_itr)
            content_dict = OrderedDict()
            visualize_indices = scores.argsort()[:10]

            # start images
            for c in range(self._n_cam):
                name = 'cam_{}_start'.format(c)
                start_img = self._images[-1, c].copy()

                for p in range(self._n_desig):
                    h, w = self._desig_pix[c, p]
                    cv2.circle(start_img,(w,h), 1, (255,0,0), -1)
                    h, w = self._goal_pix[c, p]
                    cv2.circle(start_img,(w,h), 1, (0,0,255), -1)

                save_path = save_img(self._verbose_worker, verbose_folder, name, start_img)
                content_dict[name] = [save_path for _ in visualize_indices]

            # render distributions
            for c in range(self._n_cam):
                for p in range(self._n_desig):
                    dist_p = [gen_distrib[g_i, :, c, :, :, p] for g_i in visualize_indices]
                    for v in range(len(dist_p)):
                        rendered = []
                        for t in range(gen_distrib.shape[1]):
                            dist = dist_p[v][t] / (np.amax(dist_p[v][t]) + 1e-6)
                            rendered.append((np.squeeze(plt.cm.viridis(dist)[:, :, :3]) * 255).astype(np.uint8))
                        dist_p[v] = rendered
                    desig_name = 'cam_{}_desig_{}'.format(c, p)
                    content_dict[desig_name] = save_gifs(self._verbose_worker, verbose_folder,
                                                        desig_name, dist_p)

            # render predicted images
            for c in range(self._n_cam):
                verbose_images = [(gen_images[g_i, :, c] * 255).astype(np.uint8) for g_i in visualize_indices]
                row_name = 'cam_{}_pred_images'.format(c)
                content_dict[row_name] = save_gifs(self._verbose_worker, verbose_folder,
                                                       row_name, verbose_images)

            # save scores
            content_dict['scores'] = scores[visualize_indices]
            html_page = fill_template(cem_itr, self._t, content_dict, img_height=self._hp.verbose_img_height)
            save_html(self._verbose_worker, "{}/plan.html".format(verbose_folder), html_page)

        return scores

    def _eval_pixel_cost(self, cem_itr, gen_distrib, gen_images):
        scores_per_task = []

        for icam in range(self._n_cam):
            for p in range(self._n_desig):
                distance_grid = self._get_distancegrid(self._goal_pix[icam, p])
                score = self._expected_distance(icam, p, gen_distrib[:, :, icam, :, :, p], distance_grid,
                                                normalize=True)
                
                scores_per_task.append(score)
                self._logger.log(
                    'best flow score of task {} cam{}  :{}'.format(p, icam, np.min(scores_per_task[-1])))

        scores_per_task = np.stack(scores_per_task, axis=1)

        if self._hp.only_take_first_view:
            scores_per_task = scores_per_task[:, 0][:, None]

        scores = np.mean(scores_per_task, axis=1)

        bestind = scores.argsort()[0]
        for icam in range(self._n_cam):
            for p in range(self._n_desig):
                self._logger.log('flow score of best traj for task{} cam{} :{}'.format(p, icam, scores_per_task[
                    bestind, p + icam * self._n_desig]))

        if self._hp.predictor_propagation:
            if cem_itr == (self._hp.iterations - 1):
                # pick the prop distrib from the action actually chosen after the last iteration (i.e. self.indices[0])
                bestind = scores.argsort()[0]
                self._chosen_distrib = gen_distrib[bestind]
        return scores

    def _expected_distance(self, icam, idesig, gen_distrib, distance_grid, normalize=True):
        """
        :param gen_distrib: shape [batch, t, r, c]
        :param distance_grid: shape [r, c]
        :return:
        """
        assert len(gen_distrib.shape) == 4
        t_mult = np.ones([self.predictor.sequence_length - self._net_context])
        t_mult[-1] = self._hp.finalweight

        gen_distrib = gen_distrib.copy()
        #normalize prob distributions
        if normalize:
            gen_distrib /= np.sum(np.sum(gen_distrib, axis=2), 2)[:,:, None, None]
        gen_distrib *= distance_grid[None, None]
        scores = np.sum(np.sum(gen_distrib, axis=2),2)

        scores *= t_mult[None]
        scores = np.sum(scores, axis=1)/np.sum(t_mult)
        return scores

    def _get_distancegrid(self, goal_pix):
        distance_grid = np.empty((self._img_height, self._img_width))
        for i in range(self._img_height):
            for j in range(self._img_width):
                pos = np.array([i, j])
                distance_grid[i, j] = np.linalg.norm(goal_pix - pos)

        self._logger.log('making distance grid with goal_pix', goal_pix)
        return distance_grid

    def _make_input_distrib(self, itr):
        if self._hp.predictor_propagation and self._chosen_distrib is not None:  # using the predictor's DNA to propagate, no correction
            input_distrib = self._chosen_distrib[-self._net_context:]
        else:
            input_distrib = self._switch_on_pix(self._desig_pix)
        return input_distrib
    
    def _switch_on_pix(self, desig):
        one_hot_images = np.zeros((1, self._net_context, self._n_cam, self._img_height, self._img_width, self._n_desig), dtype=np.float32)
        desig = np.clip(desig, np.zeros(2).reshape((1, 2)), np.array([self._img_height, self._img_width]).reshape((1, 2)) - 1).astype(np.int)
        # switch on pixels
        for icam in range(self._n_cam):
            for p in range(self._n_desig):
                one_hot_images[:, :, icam, desig[icam, p, 0], desig[icam, p, 1], p] = 1.
                self._logger.log('using desig pix', desig[icam, p, 0], desig[icam, p, 1])

        return one_hot_images[0]

    def act(self, t=None, i_tr=None, desig_pix=None, goal_pix=None, images=None, state=None, verbose_worker=None):
        """
        Return a random action for a state.
        Args:
            if performing highres tracking images is highres image
            t: the current controller's Time step
            goal_pix: in coordinates of small image
            desig_pix: in coordinates of small image
        """
        self._desig_pix = np.array(desig_pix).reshape((self._n_cam, self._n_desig, 2))
        self._goal_pix = np.array(goal_pix).reshape((self._n_cam, self._n_desig, 2))

        self._images = images

        self._verbose_worker = verbose_worker

        return super(PixelCostController, self).act(t, i_tr, state)

