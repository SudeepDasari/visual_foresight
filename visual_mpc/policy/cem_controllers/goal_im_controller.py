import numpy as np
import imp
from .cem_base_controller import CEMBaseController
from .visualizer.construct_html import save_gifs, save_html, save_img, fill_template
import matplotlib.pyplot as plt
from collections import OrderedDict
from visual_mpc.video_prediction.pred_util import get_context, rollout_predictions
import cv2


class GoalImController(CEMBaseController):
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

        params = imp.load_source('params', ag_params['current_dir'] + '/conf.py')
        netconf = params.configuration
        self.predictor = netconf['setup_predictor'](ag_params, netconf, gpu_id, ngpu, self._logger)

        self._net_bsize = netconf['batch_size']
        self._net_seqlen = netconf['sequence_length']

        self._net_context = netconf['context_frames']
        self._hp.start_planning = self._net_context

        self._n_desig = netconf.get('ndesig', None)
        self._img_height, self._img_width = netconf['orig_size']

        self._n_cam = netconf['ncam']

        self._desig_pix = None
        self._goal_pix = None
        self._images = None

        if self._hp.predictor_propagation:
            self._rec_input_distrib = []  # record the input distributions

    def _default_hparams(self):
        default_dict = {
            "verbose_img_height": 128,
            'predictor_propagation':False,
            'only_take_first_view':False,
            'state_append': None,
            'finalweight': 10.
        }
        parent_params = super(GoalImController, self)._default_hparams()

        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def reset(self):
        super(GoalImController, self).reset()
        if self._hp.predictor_propagation:
            self._rec_input_distrib = []  # record the input distributions

    # def switch_on_pix(self, desig):
    #     one_hot_images = np.zeros((1, self._net_context, self._n_cam, self._img_height, self._img_width, self._n_desig), dtype=np.float32)
    #     desig = np.clip(desig, np.zeros(2).reshape((1, 2)), np.array([self._img_height, self._img_width]).reshape((1, 2)) - 1).astype(np.int)
    #     # switch on pixels
    #     for icam in range(self._n_cam):
    #         for p in range(self._n_desig):
    #             one_hot_images[:, :, icam, desig[icam, p, 0], desig[icam, p, 1], p] = 1.
    #             self._logger.log('using desig pix', desig[icam, p, 0], desig[icam, p, 1])

    #     return one_hot_images

    def evaluate_rollouts(self, actions, cem_itr):
        last_frames, last_states = get_context(self._net_context, self._t,
                                               self._state, self._images, self._hp)
        # input_distrib = self._make_input_distrib(cem_itr)

        gen_images, _, _ = rollout_predictions(self.predictor, self._net_bsize, actions,
                                                      last_frames, last_states)#, input_distrib, logger=self._logger)[:2]
        gen_images = np.concatenate(gen_images, 0)
        print(gen_images.shape)

        goalim = cv2.imread('/home/sudeep/Documents/ext_data/robot_data/sawyer/goal_images/vestri/train/collection/traj1/images0/im_2.jpg')
        print(goalim.shape)
        goalim = cv2.resize(goalim, (64, 48), interpolation=cv2.INTER_AREA)
        goalim = cv2.cvtColor(goalim, cv2.COLOR_BGR2RGB)
        goalims = np.repeat(np.expand_dims(goalim, 0), 600, 0)
        print(goalims.shape)
        scores = ((gen_images[:, -1, 0, :, :, :] - goalims)**2).mean((1,2,3))
        print(scores.shape)


        # gen_distrib = np.concatenate(gen_distrib, 0)

        # scores = self._eval_pixel_cost(cem_itr, gen_distrib, gen_images)
        
        if self._verbose_condition(cem_itr):
            verbose_folder = "planning_{}_itr_{}".format(self._t, cem_itr)
            content_dict = OrderedDict()
            visualize_indices = scores.argsort()[:10]

            # start images
            for c in range(self._n_cam):
                name = 'cam_{}_start'.format(c)
                save_path = save_img(self._verbose_worker, verbose_folder, name, self._images[-1, c])
                content_dict[name] = [save_path for _ in visualize_indices]

            for c in range(self._n_cam):
                name = 'cam_{}_goal'.format(c)
                save_path = save_img(self._verbose_worker, verbose_folder, name, goalim)
                content_dict[name] = [save_path for _ in visualize_indices]

            # # render distributions
            # for c in range(self._n_cam):
            #     for p in range(self._n_desig):
            #         dist_p = [gen_distrib[g_i, :, c, :, :, p] for g_i in visualize_indices]
            #         for v in range(len(dist_p)):
            #             rendered = []
            #             for t in range(gen_distrib.shape[1]):
            #                 dist = dist_p[v][t] / (np.amax(dist_p[v][t]) + 1e-6)
            #                 rendered.append((np.squeeze(plt.cm.viridis(dist)[:, :, :3]) * 255).astype(np.uint8))
            #             dist_p[v] = rendered
            #         desig_name = 'cam_{}_desig_{}'.format(c, p)
            #         content_dict[desig_name] = save_gifs(self._verbose_worker, verbose_folder,
            #                                             desig_name, dist_p)

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

    # def _eval_pixel_cost(self, cem_itr, gen_distrib, gen_images):
    #     scores_per_task = []

    #     for icam in range(self._n_cam):
    #         for p in range(self._n_desig):
    #             distance_grid = self._get_distancegrid(self._goal_pix[icam, p])
    #             score = self._expected_distance(icam, p, gen_distrib[:, :, icam, :, :, p], distance_grid,
    #                                             normalize=True)
                
    #             scores_per_task.append(score)
    #             self._logger.log(
    #                 'best flow score of task {} cam{}  :{}'.format(p, icam, np.min(scores_per_task[-1])))

    #     scores_per_task = np.stack(scores_per_task, axis=1)

    #     if self._hp.only_take_first_view:
    #         scores_per_task = scores_per_task[:, 0][:, None]

    #     scores = np.mean(scores_per_task, axis=1)

    #     bestind = scores.argsort()[0]
    #     for icam in range(self._n_cam):
    #         for p in range(self._n_desig):
    #             self._logger.log('flow score of best traj for task{} cam{} :{}'.format(p, icam, scores_per_task[
    #                 bestind, p + icam * self._n_desig]))

    #     if self._hp.predictor_propagation:
    #         if cem_itr == (self._hp.iterations - 1):
    #             # pick the prop distrib from the action actually chosen after the last iteration (i.e. self.indices[0])
    #             bestind = scores.argsort()[0]
    #             best_gen_distrib = gen_distrib[bestind, self._net_context].reshape(1, self._n_cam, self._img_height,
    #                                                                                self._img_width, self._n_desig)
    #             self._rec_input_distrib.append(best_gen_distrib)
    #     return scores

    # def _expected_distance(self, icam, idesig, gen_distrib, distance_grid, normalize=True):
    #     """
    #     :param gen_distrib: shape [batch, t, r, c]
    #     :param distance_grid: shape [r, c]
    #     :return:
    #     """
    #     assert len(gen_distrib.shape) == 4
    #     t_mult = np.ones([self._net_seqlen - self._net_context])
    #     t_mult[-1] = self._hp.finalweight

    #     gen_distrib = gen_distrib.copy()
    #     #normalize prob distributions
    #     if normalize:
    #         gen_distrib /= np.sum(np.sum(gen_distrib, axis=2), 2)[:,:, None, None]
    #     gen_distrib *= distance_grid[None, None]
    #     scores = np.sum(np.sum(gen_distrib, axis=2),2)

    #     scores *= t_mult[None]
    #     scores = np.sum(scores, axis=1)/np.sum(t_mult)
    #     return scores

    # def _get_distancegrid(self, goal_pix):
    #     distance_grid = np.empty((self._img_height, self._img_width))
    #     for i in range(self._img_height):
    #         for j in range(self._img_width):
    #             pos = np.array([i, j])
    #             distance_grid[i, j] = np.linalg.norm(goal_pix - pos)

    #     self._logger.log('making distance grid with goal_pix', goal_pix)
    #     return distance_grid

    # def _make_input_distrib(self, itr):
    #     if self._hp.predictor_propagation:  # using the predictor's DNA to propagate, no correction
    #         input_distrib = self._get_recinput(itr, self._rec_input_distrib, self._desig_pix)
    #     else:
    #         input_distrib = self.switch_on_pix(self._desig_pix)
    #     return input_distrib

    # def _get_recinput(self, itr, rec_input_distrib, desig):
    #     ctxt = self._net_context
    #     if len(rec_input_distrib) < ctxt:
    #         input_distrib = self.switch_on_pix(desig)
    #         if itr == 0:
    #             rec_input_distrib.append(input_distrib[:, 0])
    #     else:
    #         input_distrib = [rec_input_distrib[c] for c in range(-ctxt, 0)]
    #         input_distrib = np.stack(input_distrib, axis=1)
    #     return input_distrib

    def act(self, t=None, i_tr=None, desig_pix=None, goal_pix=None, images=None, state=None, verbose_worker=None):
        """
        Return a random action for a state.
        Args:
            if performing highres tracking images is highres image
            t: the current controller's Time step
            goal_pix: in coordinates of small image
            desig_pix: in coordinates of small image
        """
        # self._desig_pix = np.array(desig_pix).reshape((self._n_cam, self._n_desig, 2))
        # self._goal_pix = np.array(goal_pix).reshape((self._n_cam, self._n_desig, 2))

        self._images = images

        self._verbose_worker = verbose_worker

        return super(GoalImController, self).act(t, i_tr, state)

