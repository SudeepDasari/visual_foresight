from .cemsampler import CEMSampler
import numpy as np
from visual_mpc.policy.utils.controller_utils import construct_initial_sigma, reuse_cov, \
    reuse_action, truncate_movement, make_blockdiagonal, discretize


class GaussianCEMSampler(CEMSampler):
    def __init__(self, hp, adim, sdim, **kwargs):
        super(GaussianCEMSampler, self).__init__(hp, adim, sdim, **kwargs)
        self._sigma, self._sigma_prev = None, None
        self._mean = None
        self._t = 0

    def sample_initial_actions(self, nsamples, current_state):
        if not self._hp.reuse_cov or self._t < self._hp.repeat - 1 or self._sigma is None:
            self._sigma = construct_initial_sigma(self._hp, self._adim, self._t)
            self._sigma_prev = self._sigma
        else:
            self._sigma = reuse_cov(self._sigma, self._adim, self._hp)

        if not self._hp.reuse_mean or self._t < self._hp.repeat - 1 or self._mean is None:
            self._mean = np.zeros(self._adim * self._hp.naction_steps)
        else:
            self._mean = reuse_action(self._chosen_actions[-1], self._hp)

        self._t += 1
        return self._sample(nsamples)

    def sample_next_actions(self, n_samples, best_actions):
        self._fit_gaussians(best_actions)
        return self._sample(n_samples)

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

    def _sample(self, M):
        if self._hp.reuse_cov or self._hp.reuse_mean:
            M = max(int(M * self._hp.reuse_factor), 1)

        if self._hp.rejection_sampling:
            return self._sample_actions_rej(M)
        return self._sample_actions(M)

    def _sample_actions(self, M):
        actions = np.random.multivariate_normal(self._mean, self._sigma, M)
        actions = actions.reshape(M, self._hp.naction_steps, self._adim)
        if self._hp.discrete_ind != None:
            actions = discretize(actions, M, self._hp.naction_steps, self._hp.discrete_ind)

        if self._hp.action_bound:
            actions = truncate_movement(actions, self._hp)
        actions = np.repeat(actions, self._hp.repeat, axis=1)

        if self._hp.add_zero_action:
            actions[0] = 0

        return actions

    def _fit_gaussians(self, actions):
        actions = actions.reshape(-1, self._hp.naction_steps, self._hp.repeat, self._adim)
        actions = actions[:, :, -1, :]  # taking only one of the repeated actions
        actions_flat = actions.reshape(-1, self._hp.naction_steps * self._adim)

        self._sigma = np.cov(actions_flat, rowvar=False, bias=False)
        if self._hp.cov_blockdiag:
            self._sigma = make_blockdiagonal(self._sigma, self._hp.naction_steps, self._adim)
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

                action_seq = action_seq.reshape(self._hp.naction_steps, self._adim)
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
            actions = discretize(actions, M, self._hp.naction_steps, self._hp.discrete_ind)
        actions = np.repeat(actions, self._hp.repeat, axis=1)

        print('max action val xy', np.max(actions[:,:,:2]))
        print('max action val z', np.max(actions[:,:,2]))
        return actions
