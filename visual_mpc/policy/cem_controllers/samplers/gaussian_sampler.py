from .cem_sampler import CEMSampler
import numpy as np
from visual_mpc.policy.utils.controller_utils import construct_initial_sigma, reuse_cov,\
    truncate_movement, make_blockdiagonal, discretize


class GaussianCEMSampler(CEMSampler):
    def __init__(self, hp, adim, sdim, **kwargs):
        super(GaussianCEMSampler, self).__init__(hp, adim, sdim, **kwargs)
        self._sigma, self._sigma_prev = None, None
        self._mean = None
        self._last_reduce = None

    def sample_initial_actions(self, t, nsamples, current_state):
        reduce_samp = False
        if not self._hp.reuse_cov or t < self._hp.repeat - 1 or self._sigma is None:
            self._sigma = construct_initial_sigma(self._hp, self._adim, t)
        else:
            reduce_samp = True
            self._sigma = reuse_cov(self._sigma, self._adim, self._hp)
        self._sigma_prev = self._sigma

        if not self._hp.reuse_mean or t < self._hp.repeat - 1 or self._mean is None:
            self._mean = np.zeros(self._adim * self._hp.nactions)
        else:
            assert self._best_action_plans[-1] is not None, "Cannot reuse mean if best actions are not logged!"
            best_action_plan = self._best_action_plans[-1][0]

            n_extra = best_action_plan.shape[0] % self._hp.repeat
            if n_extra > 0:
                zero_pad = np.zeros((self._hp.repeat - n_extra, self._adim))
                last_actions = np.concatenate((best_action_plan, zero_pad), axis=0)
            else:
                last_actions = best_action_plan
            last_actions = last_actions.reshape((-1, self._hp.repeat, self._adim))[:, 0, :]

            self._mean = np.zeros((self._hp.nactions, self._adim))
            self._mean[:last_actions.shape[0]] = last_actions
            self._mean = self._mean.flatten()

            reduce_samp = True

        self._last_reduce = reduce_samp
        return self._sample(nsamples, reduce_samp)

    def sample_next_actions(self, n_samples, best_actions, scores):
        self._fit_gaussians(best_actions)
        return self._sample(n_samples, self._last_reduce)

    @staticmethod
    def get_default_hparams():
        hparams = {
            'action_order': None,     #
            'initial_std': 0.05,  # std dev. in xy
            'initial_std_lift': 0.15,  # std dev. in xy
            'initial_std_rot': np.pi / 18,
            'initial_std_grasp': 2,
            'discrete_ind': None,
            'reuse_mean': False,
            'reduce_std_dev': 1.,  # reduce standard dev in later timesteps when reusing action
            'reuse_cov': False,
            'rejection_sampling': True,
            'cov_blockdiag': False,
            'smooth_cov': False,
            'nactions': 5,
            'repeat': 3,
            'add_zero_action': False,
            'action_bound': True,
            'reuse_factor': 0.5
        }
        return hparams

    def _sample(self, M, reduce_samp):
        if reduce_samp:
            M = max(int(M * self._hp.reuse_factor), 1)

        if self._hp.rejection_sampling:
            return self._sample_actions_rej(M)
        return self._sample_actions(M)

    def _sample_actions(self, M):
        actions = np.random.multivariate_normal(self._mean, self._sigma, M)
        actions = actions.reshape(M, self._hp.nactions, self._adim)
        if self._hp.discrete_ind != None:
            actions = discretize(actions, M, self._hp.nactions, self._hp.discrete_ind)

        if self._hp.action_bound:
            actions = truncate_movement(actions, self._hp)
        actions = np.repeat(actions, self._hp.repeat, axis=1)

        if self._hp.add_zero_action:
            actions[0] = 0

        return actions

    def _fit_gaussians(self, actions):
        actions = actions.reshape(-1, self._hp.nactions, self._hp.repeat, self._adim)
        actions = actions[:, :, -1, :]  # taking only one of the repeated actions
        actions_flat = actions.reshape(-1, self._hp.nactions * self._adim)

        self._sigma = np.cov(actions_flat, rowvar=False, bias=False)
        if self._hp.cov_blockdiag:
            self._sigma = make_blockdiagonal(self._sigma, self._hp.nactions, self._adim)
        if self._hp.smooth_cov:
            self._sigma = 0.5 * self._sigma + 0.5 * self._sigma_prev
            self._sigma_prev = self._sigma
        self._mean = np.mean(actions_flat, axis=0)

    def _sample_actions_rej(self, M):
        """
        Perform rejection sampling
        :return:
        """
        runs = []
        actions = []

        for i in range(M):
            ok = False
            i = 0
            while not ok:
                i +=1
                action_seq = np.random.multivariate_normal(self._mean, self._sigma, 1)

                action_seq = action_seq.reshape(self._hp.nactions, self._adim)
                xy_std = self._hp.initial_std
                lift_std = self._hp.initial_std_lift

                std_fac = 1.5
                if np.any(action_seq[:, :2] > xy_std*std_fac) or \
                        np.any(action_seq[:, :2] < -xy_std*std_fac) or \
                        np.any(action_seq[:, 2] > lift_std*std_fac) or \
                        np.any(action_seq[:, 2] < -lift_std*std_fac):
                    ok = False
                else: ok = True

            runs.append(i)
            actions.append(action_seq)
        actions = np.stack(actions, axis=0)

        if self._hp.stochastic_planning:
            actions = np.repeat(actions,self._hp.stochastic_planning[0], 0)

        print('rejection smp max trials', max(runs))
        if self._hp.discrete_ind != None:
            actions = discretize(actions, M, self._hp.nactions, self._hp.discrete_ind)
        actions = np.repeat(actions, self._hp.repeat, axis=1)

        print('max action val xy', np.max(actions[:,:,:2]))
        print('max action val z', np.max(actions[:,:,2]))
        return actions
