import numpy as np
import imp
from .visualizer.make_cem_visuals import CEM_Visual_Preparation
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import copy

from threading import Thread
import sys
if sys.version_info[0] == 2:
    from Queue import Queue
else:
    from queue import Queue

import time
from visual_mpc.policy.utils.controller_utils import save_track_pkl
from .cem_controller_base import CEM_Controller_Base


verbose_queue = Queue()
def verbose_worker():
    req = 0
    while True:
        print('servicing req', req)
        try:
            plt.switch_backend('Agg')
            visualizer, vd = verbose_queue.get(True)
            visualizer.visualize(vd)
        except RuntimeError:
            print("TKINTER ERROR, SKIPPING")
        req += 1

class VisualzationData():
    def __init__(self):
        """
        container for visualization data
        """
        pass


class CEM_Controller_Vidpred(CEM_Controller_Base):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        """
        :param ag_params:
        :param policyparams:
        :param predictor:
        :param save_subdir:
        :param gdnet: goal-distance network
        """

        self._hp = self._default_hparams()
        self.override_defaults(policyparams)

        CEM_Controller_Base.__init__(self, ag_params, policyparams)

        params = imp.load_source('params', ag_params['current_dir'] + '/conf.py')
        self.netconf = params.configuration

        if ngpu > 1 and self._hp.extra_score_functions is not None:
            vpred_ngpu = ngpu - 1
        else: vpred_ngpu = ngpu

        self.predictor = self.netconf['setup_predictor'](ag_params, self.netconf, gpu_id, vpred_ngpu, self.logger)

        self.bsize = self.netconf['batch_size']

        self.seqlen = self.netconf['sequence_length']

        # override params here:
        if 'num_samples' not in policyparams:
            self.M = self.bsize

        assert self.naction_steps * self.repeat == self.seqlen
        assert self.len_pred == self.seqlen - self.ncontxt

        self.ncontxt = self.netconf['context_frames']

        if 'ndesig' in self.netconf:        # total number of
            self.ndesig = self.netconf['ndesig']
        else: self.ndesig = None

        self.img_height, self.img_width = self.netconf['orig_size']

        self.ncam = self.netconf['ncam']

        self.desig_pix = None
        self.goal_mask = None
        self.goal_pix = None

        if self._hp.predictor_propagation:
            self.rec_input_distrib = []  # record the input distributions

        self.parallel_vis = True
        if self.parallel_vis:
            self._thread = Thread(target=verbose_worker)
            self._thread.start()
        self.goal_image = None

        self.best_cost_perstep = np.zeros([self.ncam, self.ndesig, self.seqlen])

        self.ntask = self.ndesig  # will be overwritten in derived classes
        self.vd = VisualzationData()
        self.visualizer = CEM_Visual_Preparation()

        self.reg_tradeoff = np.ones([self.ncam, self.ndesig])/self.ncam/self.ndesig

        if self._hp.extra_score_functions is not None:
            funcs = []
            for func, params in self._hp.extra_score_functions:
                if 'device_id' in params:
                    params['device_id'] = gpu_id + ngpu - 1
                funcs.append(func(**params))
            self._hp.extra_score_functions = funcs

    def _setup_visualizer(self, default=CEM_Visual_Preparation):
        run_freq = 1
        if self._hp.verbose_every_itr:
            run_freq = 3
        self.visualizer = self.policyparams.get('visualizer', default)(run_freq)

    def _default_hparams(self):
        default_dict = {
            'predictor_propagation':False,
            'trade_off_reg':True,
            'only_take_first_view':False,
            'extra_score_functions': None,   # None if no extra, else list of (class, params) tuples
            'pixel_score_weight': 1.,
            'extra_score_weight': 1.,
            'state_append': None
        }
        parent_params = super(CEM_Controller_Vidpred, self)._default_hparams()

        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def reset(self):
        super(CEM_Controller_Vidpred, self).reset()
        if self._hp.predictor_propagation:
            self.rec_input_distrib = []  # record the input distributions

    def calc_action_cost(self, actions):
        actions_costs = np.zeros(self.M)
        for smp in range(self.M):
            force_magnitudes = np.array([np.linalg.norm(actions[smp, t]) for
                                         t in range(self.naction_steps * self.repeat)])
            actions_costs[smp]=np.sum(np.square(force_magnitudes)) * self._hp.action_cost_factor
        return actions_costs

    def switch_on_pix(self, desig):
        one_hot_images = np.zeros((1, self.netconf['context_frames'], self.ncam, self.img_height, self.img_width, self.ndesig), dtype=np.float32)
        desig = np.clip(desig, np.zeros(2).reshape((1, 2)), np.array([self.img_height, self.img_width]).reshape((1, 2)) - 1).astype(np.int)
        # switch on pixels
        for icam in range(self.ncam):
            for p in range(self.ndesig):
                one_hot_images[:, :, icam, desig[icam, p, 0], desig[icam, p, 1], p] = 1.
                self.logger.log('using desig pix',desig[icam, p, 0], desig[icam, p, 1])

        return one_hot_images

    def get_rollouts(self, actions, cem_itr, itr_times, n_samps=None):
        if n_samps is None:
            n_samps = self.M


        actions, last_frames, last_states, t_0 = self.prep_vidpred_inp(actions, cem_itr)
        input_distrib = self.make_input_distrib(cem_itr)

        t_startpred = time.time()
        if n_samps > self.bsize:
            nruns = n_samps//self.bsize
            assert self.bsize*nruns == n_samps, "bsize: {}, nruns {}, but n_samps is {}".format(self.bsize, nruns, n_samps)
        else:
            nruns = 1
            assert n_samps == self.bsize
        gen_images_l, gen_distrib_l, gen_states_l = [], [], []
        itr_times['pre_run'] = time.time() - t_0
        for run in range(nruns):
            self.logger.log('run{}'.format(run))
            t_run_loop = time.time()
            actions_ = actions[run*self.bsize:(run+1)*self.bsize]

            gen_images, gen_distrib, gen_states, _ = self.predictor(input_images=last_frames,
                                                                    input_state=last_states,
                                                                    input_actions=actions_,
                                                                    input_one_hot_images=input_distrib)
            gen_images_l.append(gen_images)
            gen_distrib_l.append(gen_distrib)
            gen_states_l.append(gen_states)
            itr_times['run{}'.format(run)] = time.time() - t_run_loop
        t_run_post = time.time()
        gen_images = np.concatenate(gen_images_l, 0)
        gen_distrib = np.concatenate(gen_distrib_l, 0)
        if gen_states_l[0] is not None:
            gen_states = np.concatenate(gen_states_l, 0)
        itr_times['t_concat'] = time.time() - t_run_post
        self.logger.log('time for videoprediction {}'.format(time.time() - t_startpred))
        t_run_post = time.time()

        scores = self.eval_planningcost(cem_itr, gen_distrib, gen_images)

        itr_times['run_post'] = time.time() - t_run_post
        tstart_verbose = time.time()

        self.vd.t = self.t
        self.vd.scores = scores
        self.vd.agentparams = self.agentparams
        self.vd.hp = self._hp
        self.vd.netconf = self.netconf
        self.vd.ndesig = self.ndesig
        self.vd.gen_distrib = gen_distrib
        self.vd.gen_images = gen_images
        self.vd.K = self.K
        self.vd.cem_itr = cem_itr
        self.vd.last_frames = last_frames
        self.vd.goal_pix = self.goal_pix
        self.vd.ncam = self.ncam
        self.vd.image_height = self.img_height

        if self.verbose and cem_itr == self._hp.iterations-1 and self.i_tr % self.verbose_freq ==0 or \
                (self._hp.verbose_every_itr and self.i_tr % self.verbose_freq ==0):
            if self.parallel_vis:
                print('t{} cemitr {}'.format(self.t, cem_itr))
                verbose_queue.put((self.visualizer, copy.deepcopy(self.vd)))
            else:
                self.visualizer.visualize(self.vd)

        if 'save_desig_pos' in self.agentparams:
            save_track_pkl(self, self.t, cem_itr)

        itr_times['verbose_time'] = time.time() - tstart_verbose
        self.logger.log('verbose time', time.time() - tstart_verbose)

        return scores

    def eval_planningcost(self, cem_itr, gen_distrib, gen_images):

        t_startcalcscores = time.time()
        scores_per_task = []

        for icam in range(self.ncam):
            for p in range(self.ndesig):
                distance_grid = self.get_distancegrid(self.goal_pix[icam, p])
                score = self.calc_scores(icam, p, gen_distrib[:, :, icam, :, :, p], distance_grid,
                                         normalize=True)
                if self._hp.trade_off_reg:
                    score *= self.reg_tradeoff[icam, p]
                scores_per_task.append(score)
                self.logger.log(
                    'best flow score of task {} cam{}  :{}'.format(p, icam, np.min(scores_per_task[-1])))

        scores_per_task = np.stack(scores_per_task, axis=1)

        if self._hp.only_take_first_view:
            scores_per_task = scores_per_task[:, 0][:, None]

        scores = np.mean(scores_per_task, axis=1)

        if self._hp.extra_score_functions is not None:
            batches, T, ncams, height, width, channels = gen_images.shape
            extra_costs = np.zeros((batches))
            score_std, score_mean = np.std(scores), np.mean(scores)

            for cost in self._hp.extra_score_functions:
                c_score = np.zeros((batches, T))
                for t in range(T):
                    if self.goal_image is None:
                        c_score[:, t] = cost.score(images=gen_images[:, t])
                    else:
                        goal_feed = self.goal_image[-1]
                        if len(goal_feed.shape) == 4:
                            goal_feed = goal_feed[None]

                        c_score[:, t] = cost.score(images=gen_images[:, t], goal_images=goal_feed)
                c_score[:, -1] *= self._hp.finalweight
                c_score = np.sum(c_score, 1)
                if score_std > 0 and score_mean > 0 and self._hp.pixel_score_weight > 0:
                    old_std, old_mean = np.std(c_score), np.mean(c_score)
                    c_score = score_std * ((c_score - old_mean) / old_std) + score_mean
                extra_costs += c_score

            print('best extra costs: {}'.format(np.amin(extra_costs)))
            scores = self._hp.pixel_score_weight * scores + self._hp.extra_score_weight * extra_costs

        bestind = scores.argsort()[0]
        for icam in range(self.ncam):
            for p in range(self.ndesig):
                self.logger.log('flow score of best traj for task{} cam{} :{}'.format(p, icam, scores_per_task[
                    bestind, p + icam * self.ndesig]))

        self.best_cost_perstep = self.cost_perstep[bestind]

        if self._hp.predictor_propagation:
            if cem_itr == (self._hp.iterations - 1):
                # pick the prop distrib from the action actually chosen after the last iteration (i.e. self.indices[0])
                bestind = scores.argsort()[0]
                best_gen_distrib = gen_distrib[bestind, self.ncontxt].reshape(1, self.ncam, self.img_height,
                                                                              self.img_width, self.ndesig)
                self.rec_input_distrib.append(best_gen_distrib)
        self.logger.log('time to calc scores {}'.format(time.time() - t_startcalcscores))
        return scores

    def prep_vidpred_inp(self, actions, cem_itr):
        t_0 = time.time()
        ctxt = self.netconf['context_frames']
        last_frames = self.images[self.t - ctxt + 1:self.t + 1]  # same as [t - 1:t + 1] for context 2
        last_frames = last_frames.astype(np.float32, copy=False) / 255.
        last_frames = last_frames[None]
        last_states = self.state[self.t - ctxt + 1:self.t + 1]
        last_states = last_states[None]
        if self._hp.state_append:
            last_state_append = np.tile(np.array([[self._hp.state_append]]), (1, ctxt, 1))
            last_states = np.concatenate((last_states, last_state_append), -1)

        self.logger.log('t0 ', time.time() - t_0)
        return actions, last_frames, last_states, t_0

    def calc_scores(self, icam, idesig, gen_distrib, distance_grid, normalize=True):
        """
        :param gen_distrib: shape [batch, t, r, c]
        :param distance_grid: shape [r, c]
        :return:
        """
        assert len(gen_distrib.shape) == 4
        t_mult = np.ones([self.seqlen - self.netconf['context_frames']])
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
        distance_grid = np.empty((self.img_height, self.img_width))
        for i in range(self.img_height):
            for j in range(self.img_width):
                pos = np.array([i, j])
                distance_grid[i, j] = np.linalg.norm(goal_pix - pos)

        self.logger.log('making distance grid with goal_pix', goal_pix)
        # plt.imshow(distance_grid, zorder=0, cmap=plt.get_cmap('jet'), interpolation='none')
        # plt.show()
        return distance_grid

    def make_input_distrib(self, itr):
        if self._hp.predictor_propagation:  # using the predictor's DNA to propagate, no correction
            input_distrib = self.get_recinput(itr, self.rec_input_distrib, self.desig_pix)
        else:
            input_distrib = self.switch_on_pix(self.desig_pix)
        return input_distrib

    def get_recinput(self, itr, rec_input_distrib, desig):
        ctxt = self.netconf['context_frames']
        if len(rec_input_distrib) < ctxt:
            input_distrib = self.switch_on_pix(desig)
            if itr == 0:
                rec_input_distrib.append(input_distrib[:, 0])
        else:
            input_distrib = [rec_input_distrib[c] for c in range(-ctxt, 0)]
            input_distrib = np.stack(input_distrib, axis=1)
        return input_distrib


    def act(self, t=None, i_tr=None, desig_pix=None, goal_pix=None, images=None, state=None):
        """
        Return a random action for a state.
        Args:
            if performing highres tracking images is highres image
            t: the current controller's Time step
            goal_pix: in coordinates of small image
            desig_pix: in coordinates of small image
        """
        self.desig_pix = np.array(desig_pix).reshape((self.ncam, self.ntask, 2))
        self.goal_pix = np.array(goal_pix).reshape((self.ncam, self.ntask, 2))


        self.images = images
        self.state = state
        return super(CEM_Controller_Vidpred, self).act(t, i_tr)
