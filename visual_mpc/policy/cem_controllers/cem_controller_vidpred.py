import numpy as np
import imp
from .cem_controller_base import CEM_Controller_Base
from .visualizer.construct_html import save_gifs, save_html, fill_template
import matplotlib.pyplot as plt
from collections import OrderedDict


class CEM_Controller_Vidpred(CEM_Controller_Base):
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
        CEM_Controller_Base.__init__(self, ag_params, policyparams)

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

        if self._hp.predictor_propagation:
            self._rec_input_distrib = []  # record the input distributions

    def _default_hparams(self):
        default_dict = {
            'verbose': True,
            'verbose_every_iter': False,
            "verbose_img_height": 128,
            'predictor_propagation':False,
            'only_take_first_view':False,
            'state_append': None
        }
        parent_params = super(CEM_Controller_Vidpred, self)._default_hparams()

        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def reset(self):
        super(CEM_Controller_Vidpred, self).reset()
        if self._hp.predictor_propagation:
            self._rec_input_distrib = []  # record the input distributions

    def switch_on_pix(self, desig):
        one_hot_images = np.zeros((1, self._net_context, self._n_cam, self._img_height, self._img_width, self._n_desig), dtype=np.float32)
        desig = np.clip(desig, np.zeros(2).reshape((1, 2)), np.array([self._img_height, self._img_width]).reshape((1, 2)) - 1).astype(np.int)
        # switch on pixels
        for icam in range(self._n_cam):
            for p in range(self._n_desig):
                one_hot_images[:, :, icam, desig[icam, p, 0], desig[icam, p, 1], p] = 1.
                self._logger.log('using desig pix', desig[icam, p, 0], desig[icam, p, 1])

        return one_hot_images

    def evaluate_rollouts(self, actions, cem_itr, n_samps=None):
        if n_samps is None:
            n_samps = self.M

        actions, last_frames, last_states = self._prep_vidpred_inp(actions, cem_itr)
        input_distrib = self.make_input_distrib(cem_itr)

        if n_samps > self._net_bsize:
            nruns = n_samps//self._net_bsize
            assert self._net_bsize * nruns == n_samps, "bsize: {}, nruns {}, but n_samps is {}".format(self._net_bsize, nruns, n_samps)
        else:
            nruns = 1
            assert n_samps == self._net_bsize
        gen_images_l, gen_distrib_l = [], []
        for run in range(nruns):
            self._logger.log('run{}'.format(run))
            actions_ = actions[run*self._net_bsize:(run + 1) * self._net_bsize]

            gen_images, gen_distrib, _, _ = self.predictor(input_images=last_frames,
                                                                    input_state=last_states,
                                                                    input_actions=actions_,
                                                                    input_one_hot_images=input_distrib)
            gen_images_l.append(gen_images)
            gen_distrib_l.append(gen_distrib)

        gen_images = np.concatenate(gen_images_l, 0)
        gen_distrib = np.concatenate(gen_distrib_l, 0)

        scores = self.eval_planningcost(cem_itr, gen_distrib, gen_images)

        if self._hp.verbose:
            if self._hp.verbose_every_iter or cem_itr == self._n_iter - 1:
                verbose_folder = "planning_{}_itr_{}".format(self._t, cem_itr)
                content_dict = OrderedDict()
                visualize_indices = scores.argsort()[:10]

                # render distributions
                for p in range(self._n_desig):
                    dist_p = [gen_distrib[g_i, :, :, :, p] for g_i in visualize_indices]
                    for v in range(len(dist_p)):
                        rendered = []
                        for t in range(gen_distrib.shape[1]):
                            dist = dist_p[v][t] / (np.amax(dist_p[v][t]) + 1e-6)
                            rendered.append((np.squeeze(plt.cm.viridis(dist)[:, :, :3]) * 255).astype(np.uint8))
                        dist_p[v] = rendered
                    desig_name = 'gen_dist_desig_{}'.format(p)
                    content_dict[desig_name] = save_gifs(self._verbose_worker, verbose_folder,
                                                        desig_name, dist_p)

                # render predicted images
                verbose_images = [(gen_images[g_i] * 255).astype(np.uint8) for g_i in visualize_indices]
                content_dict['gen_images'] = save_gifs(self._verbose_worker, verbose_folder,
                                                       'gen_images', verbose_images)

                # save scores
                content_dict['scores'] = scores[visualize_indices]

                html_page = fill_template(cem_itr, self._t, content_dict, img_height=self._hp.verbose_img_height)
                save_html(self._verbose_worker, "{}/plan.html".format(verbose_folder), html_page)

        return scores

    def eval_planningcost(self, cem_itr, gen_distrib, gen_images):
        scores_per_task = []

        for icam in range(self._n_cam):
            for p in range(self._n_desig):
                distance_grid = self.get_distancegrid(self._goal_pix[icam, p])
                score = self.calc_scores(icam, p, gen_distrib[:, :, icam, :, :, p], distance_grid,
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
                best_gen_distrib = gen_distrib[bestind, self._net_context].reshape(1, self._n_cam, self._img_height,
                                                                                   self._img_width, self._n_desig)
                self._rec_input_distrib.append(best_gen_distrib)
        return scores

    def _prep_vidpred_inp(self, actions, cem_itr):
        ctxt = self._net_context
        last_frames = self.images[self._t - ctxt + 1:self._t + 1]  # same as [t - 1:t + 1] for context 2
        last_frames = last_frames.astype(np.float32) / 255.
        last_frames = last_frames[None]
        last_states = self.state[self._t - ctxt + 1:self._t + 1]
        last_states = last_states[None]
        if self._hp.state_append:
            last_state_append = np.tile(np.array([[self._hp.state_append]]), (1, ctxt, 1))
            last_states = np.concatenate((last_states, last_state_append), -1)

        return actions, last_frames, last_states

    def calc_scores(self, icam, idesig, gen_distrib, distance_grid, normalize=True):
        """
        :param gen_distrib: shape [batch, t, r, c]
        :param distance_grid: shape [r, c]
        :return:
        """
        assert len(gen_distrib.shape) == 4
        t_mult = np.ones([self._net_seqlen - self._net_context])
        t_mult[-1] = self._hp.finalweight

        gen_distrib = gen_distrib.copy()
        #normalize prob distributions
        if normalize:
            gen_distrib /= np.sum(np.sum(gen_distrib, axis=2), 2)[:,:, None, None]
        gen_distrib *= distance_grid[None, None]
        scores = np.sum(np.sum(gen_distrib, axis=2),2)
        self.cost_perstep[:,icam, idesig] = scores
        scores *= t_mult[None]
        scores = np.sum(scores, axis=1)/np.sum(t_mult)
        return scores

    def get_distancegrid(self, goal_pix):
        distance_grid = np.empty((self._img_height, self._img_width))
        for i in range(self._img_height):
            for j in range(self._img_width):
                pos = np.array([i, j])
                distance_grid[i, j] = np.linalg.norm(goal_pix - pos)

        self._logger.log('making distance grid with goal_pix', goal_pix)
        return distance_grid

    def make_input_distrib(self, itr):
        if self._hp.predictor_propagation:  # using the predictor's DNA to propagate, no correction
            input_distrib = self.get_recinput(itr, self._rec_input_distrib, self._desig_pix)
        else:
            input_distrib = self.switch_on_pix(self._desig_pix)
        return input_distrib

    def get_recinput(self, itr, rec_input_distrib, desig):
        ctxt = self._net_context
        if len(rec_input_distrib) < ctxt:
            input_distrib = self.switch_on_pix(desig)
            if itr == 0:
                rec_input_distrib.append(input_distrib[:, 0])
        else:
            input_distrib = [rec_input_distrib[c] for c in range(-ctxt, 0)]
            input_distrib = np.stack(input_distrib, axis=1)
        return input_distrib


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

        self.images = images
        self.state = state

        self._verbose_worker = verbose_worker

        return super(CEM_Controller_Vidpred, self).act(t, i_tr, state)

