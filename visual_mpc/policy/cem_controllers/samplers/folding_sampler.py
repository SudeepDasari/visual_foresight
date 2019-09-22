import copy
import numpy as np
from .cem_sampler import CEMSampler
from visual_mpc.policy.utils.controller_utils import construct_initial_sigma


class FoldingCEMSampler(CEMSampler):
    def __init__(self, hp, adim, sdim, **kwargs):
        super(FoldingCEMSampler, self).__init__(hp, adim, sdim, **kwargs)
        assert adim == 4, "Requires base action dimension of 4"
        assert hp.nactions >= 5, "Requires at least 5 steps"

       
        self._repeat = hp.repeat
        self._steps = hp.nactions
        self._base_mean, self._full_sigma, self._base_sigma = None, None, None

    def sample_initial_actions(self, t, n_samples, current_state):
        """
        Samples initial actions for CEM iterations
        :param nsamples: number of samples
        :param current_state: Current state of robot
        :return: action samples in (B, T, adim) format
        """
        base_mean = np.zeros((self._steps * self._adim))
        base_sigma = construct_initial_sigma(self._hp, self._adim, t)

        self._current_state = current_state[:2]
        return self._sample(True, n_samples, base_mean, base_sigma)

    def sample_next_actions(self, n_samples, best_actions, scores):
        """
        Samples actions for CEM iterations, given BEST last actions
        :param nsamples: number of samples
        :param best_actions: number of samples
        :return: action samples in (B, T, adim) format
        """

        actions = best_actions.reshape(-1, self._hp.nactions, self._hp.repeat, self._adim)
        actions = actions[:, :, -1, :]  # taking only one of the repeated actions
        actions_flat = actions.reshape(-1, self._hp.nactions * self._adim)

        sigma = np.cov(actions_flat, rowvar=False, bias=False)
        mean = np.mean(actions_flat, axis=0)

        return self._sample(True, n_samples, mean, sigma)

    def _sample(self, is_first_itr, M, new_mean, new_sigma):
        
        self._base_mean = copy.deepcopy(new_mean)
        self._full_sigma = copy.deepcopy(new_sigma)
        self._base_sigma = self._full_sigma[:4, :4]

        assert M % 3 == 0, "splits samples into setting with 3 means"
        ret_actions = np.zeros((M, self._steps, self._adim))
        per_split = int((M * self._hp.split_frac) / 2)

        if is_first_itr:
            per_split = max(int(per_split / 2), 1)

        lower_sigma = copy.deepcopy(self._base_sigma)
        lower_sigma[:2, :2] /= 10
        lower_sigma[3, 3] /= 2

        for i in range(per_split):
            first_pnt, second_pnt = np.random.uniform(size=2), np.random.uniform(size=2)

            delta_first, delta_second = (first_pnt - self._current_state) / self._repeat, \
                                        (second_pnt - first_pnt) / self._repeat

            mean = np.array([delta_first[0], delta_first[1], 1, 0.])
            ret_actions[i, 0] = np.random.multivariate_normal(mean, self._base_sigma, 1).reshape(-1)

            mean = np.array([0, 0., -1, 0])
            ret_actions[i, 1] = np.random.multivariate_normal(mean, lower_sigma, 1).reshape(-1)

            mean = np.array([0, 0., 1, 0])
            ret_actions[i, 2] = np.random.multivariate_normal(mean, lower_sigma, 1).reshape(-1)

            mean = np.array([delta_second[0], delta_second[1], 1, 0])
            ret_actions[i, 3] = np.random.multivariate_normal(mean, self._base_sigma, 1).reshape(-1)

            mean = np.array([0, 0., -1, 0])
            ret_actions[i, 4] = np.random.multivariate_normal(mean, lower_sigma, 1).reshape(-1)

            if self._steps > 5:
                ret_actions[i, self._steps:] = np.random.multivariate_normal(np.zeros(4), self._base_sigma,
                                                                             self._steps - 5)

        for i in range(per_split, 2 * per_split):
            second_pnt = np.random.uniform(size=2)

            delta_second = (second_pnt - self._current_state) / self._repeat

            mean = np.array([0, 0, 1, 0.])
            ret_actions[i, 0] = np.random.multivariate_normal(mean, lower_sigma, 1).reshape(-1)

            mean = np.array([delta_second[0], delta_second[1], 1, 0])
            ret_actions[i, 1] = np.random.multivariate_normal(mean, self._base_sigma, 1).reshape(-1)

            mean = np.array([0, 0., -1, 0])
            ret_actions[i, 2] = np.random.multivariate_normal(mean, lower_sigma, 1).reshape(-1)

            mean = np.array([0, 0., 0, 0])
            ret_actions[i, 3:] = np.random.multivariate_normal(mean, lower_sigma, 1).reshape(-1)

            if self._steps > 5:
                ret_actions[i, self._steps:] = np.random.multivariate_normal(np.zeros(4), self._base_sigma,
                                                                             self._steps - 5)
        n_def_samples = ret_actions[2 * per_split:].shape[0]
        default_actions = np.random.multivariate_normal(self._base_mean, self._full_sigma, n_def_samples)
        ret_actions[2 * per_split:] = default_actions.reshape((n_def_samples, self._steps, self._adim))

        ret_actions[:, :, :3] = np.clip(ret_actions[:, :, :3], -np.array(self._hp.max_shift),
                                        np.array(self._hp.max_shift))

        return np.repeat(ret_actions, self._repeat, axis=1)

    @staticmethod
    def get_default_hparams():
        hparams_dict = {
            'action_order': None,     #
            'initial_std': 0.05,  # std dev. in xy
            'initial_std_lift': 0.15,  # std dev. in xy
            'initial_std_rot': np.pi / 18,
            'initial_std_grasp': 2,
            'nactions': 5,
            'repeat': 3,
            'max_shift': [1. / 5, 1. / 5, 1. / 3],
            'split_frac': 0.5
        }
        return hparams_dict
