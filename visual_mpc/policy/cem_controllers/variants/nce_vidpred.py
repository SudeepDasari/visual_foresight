from visual_mpc.policy.cem_controllers.cem_controller_base import CEM_Controller_Base
import numpy as np
import imp
from visual_mpc.policy.cem_controllers.visualizer.make_cem_visuals import CEM_Visual_Preparation
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import time
import control_embedding

from threading import Thread
import sys
if sys.version_info[0] == 2:
    from Queue import Queue
else:
    from queue import Queue


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


class CEM_NCE_Vidpred(CEM_Controller_Base):
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

        if ngpu > 1:
            vpred_ngpu = ngpu - 1
        else: vpred_ngpu = ngpu

        self.predictor = self.netconf['setup_predictor'](ag_params, self.netconf, gpu_id, vpred_ngpu, self.logger)
        self._scoring_func = control_embedding.deploy_model(self._hp.nce_conf_path, batch_size=self._hp.nce_batch_size,
                                                            restore_path=self._hp.nce_restore_path,
                                                            device_id=gpu_id + ngpu - 1)

        self.bsize = self.netconf['batch_size']
        self.seqlen = self.netconf['sequence_length']

        # override params here:
        if 'num_samples' not in policyparams:
            self.M = self.bsize

        assert self.naction_steps * self.repeat == self.seqlen
        assert self.len_pred == self.seqlen - self.ncontxt

        self.ncontxt = self.netconf['context_frames']

        self.img_height, self.img_width = self.netconf['orig_size']

        self.ncam = self.netconf['ncam']

        self.parallel_vis = True
        if self.parallel_vis:
            self._thread = Thread(target=verbose_worker)
            self._thread.start()

        self.images = None
        self.goal_image = None
        self.start_image = None

        self.best_cost_perstep = np.zeros([self.ncam, 1, self.seqlen])

        self.vd = VisualzationData()
        self.visualizer = CEM_Visual_Preparation()

    def _default_hparams(self):
        default_dict = {
            'nce_conf_path': '',
            'nce_restore_path': '',
            'nce_batch_size': 200,
            'state_append': None
        }
        parent_params = super(CEM_NCE_Vidpred, self)._default_hparams()

        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def get_rollouts(self, actions, cem_itr, itr_times, n_samps=None):
        if n_samps is None:
            n_samps = self.M

        actions, last_frames, last_states, t_0 = self.prep_vidpred_inp(actions, cem_itr)

        t_startpred = time.time()
        if n_samps > self.bsize:
            nruns = n_samps//self.bsize
            assert self.bsize*nruns == n_samps, "bsize: {}, nruns {}, but n_samps is {}".format(self.bsize, nruns, n_samps)
        else:
            nruns = 1
            assert n_samps == self.bsize
        gen_images_l, gen_states_l = [], []
        itr_times['pre_run'] = time.time() - t_0
        for run in range(nruns):
            self.logger.log('run{}'.format(run))
            t_run_loop = time.time()
            actions_ = actions[run*self.bsize:(run+1)*self.bsize]

            gen_images, _, gen_states, _ = self.predictor(input_images=last_frames,
                                                          input_state=last_states,
                                                          input_actions=actions_)
            gen_images_l.append(gen_images)
            gen_states_l.append(gen_states)
            itr_times['run{}'.format(run)] = time.time() - t_run_loop

        t_run_post = time.time()
        gen_images = np.concatenate(gen_images_l, 0)
        if gen_states_l[0] is not None:
            gen_states = np.concatenate(gen_states_l, 0)

        itr_times['t_concat'] = time.time() - t_run_post
        self.logger.log('time for videoprediction {}'.format(time.time() - t_startpred))
        t_run_post = time.time()

        scores = self.eval_planningcost(gen_images)

        itr_times['run_post'] = time.time() - t_run_post

        self.vd.t = self.t
        self.vd.scores = scores
        self.vd.agentparams = self.agentparams
        self.vd.hp = self._hp
        self.vd.netconf = self.netconf
        self.vd.ndesig = self.ndesig
        self.vd.gen_images = gen_images
        self.vd.K = self.K
        self.vd.cem_itr = cem_itr
        self.vd.last_frames = last_frames
        self.vd.ncam = self.ncam
        self.vd.image_height = self.img_height

        return scores

    def eval_planningcost(self, gen_images):
        b_size, n_pred, n_cam, height, width, channels = gen_images.shape
        scores = np.zeros((n_cam, b_size, n_pred))

        for c in range(n_cam):
            goal, start = self.goal_image[c][None], self.start_image[c][None]
            input_images = gen_images[:, :, c].reshape((b_size * n_pred, height, width, channels))
            embed_dict = self._scoring_func(goal * 255, start * 255, input_images * 255)

            gs_enc, in_enc = embed_dict['goal_enc'][0][None], embed_dict['input_enc'].reshape((b_size, n_pred, -1))
            scores[c] = np.matmul(gs_enc[None], np.swapaxes(in_enc, 2, 1))[:, 0]

        scores = np.sum(-scores, axis=0)
        scores[:, -1] *= self._hp.finalweight
        return np.sum(scores, axis=-1)

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

    def act(self, t=None, i_tr=None, goal_image=None, images=None, state=None):
        """
        Return a random action for a state.
        Args:
            if performing highres tracking images is highres image
            t: the current controller's Time step
            goal_pix: in coordinates of small image
            desig_pix: in coordinates of small image
        """
        self.start_image = goal_image[0]
        self.goal_image = goal_image[1]
        self.images = images
        self.state = state

        return super(CEM_NCE_Vidpred, self).act(t, i_tr)
