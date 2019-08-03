"""
TODO: add paper cite
"""

import numpy as np
from .cem_sampler import CEMSampler
import matplotlib.pyplot as plt


class CorrelatedNoiseSampler(CEMSampler):
    def __init__(self, hp, adim, sdim, **kwargs):
        self._hp = hp
        self._adim, self._sdim = len(self._hp.initial_std), sdim
        self._chosen_actions = []
        self._best_action_plans = []
    
    def _sample_noise(self, n_samples, cov=None):
        noise = np.random.normal(size=(n_samples, self._hp.nactions, self._adim))
        if self._hp.mean_bias is not None:
            mean_bias = self._hp.mean_bias
            print('mean bias', mean_bias)
        else: mean_bias = 0.

        if cov is None:
            noise = noise * np.array(self._hp.initial_std).reshape((1, 1, -1)) + mean_bias[None, None]

        else:
            noise = np.matmul(noise.reshape((n_samples, -1)), cov).reshape((n_samples, self._hp.nactions, self._adim))

        final_actions = noise.copy()
        for i in range(self._hp.nactions):
            if self._hp.smooth_across_last_action and i == 0 and len(self._chosen_actions):
                final_actions[:, i, :] = self._hp.beta_0 * noise[:, i, :] + self._hp.beta_1 * self._hp._chosen_actions[-1][None]
            else:
                final_actions[:, i, :] = self._hp.beta_0 * noise[:, i, :] + self._hp.beta_1 * final_actions[:, i - 1, :]
        
        return final_actions

    def sample_initial_actions(self, t, n_samples, current_state):
        """
        Samples initial actions for CEM iterations
        :param nsamples: number of samples
        :param current_state: Current state of robot
        :return: action samples in (B, T, adim) format
        """
        return self._sample_noise(n_samples)
        

    def sample_next_actions(self, n_samples, best_actions, scores):
        """
        Samples actions for CEM iterations, given BEST last actions
        :param nsamples: number of samples
        :param best_actions: number of samples
        :return: action samples in (B, T, adim) format
        """
        rewards = -scores
        S = np.exp(self._hp.kappa * (rewards - np.max(rewards)))

        weighted_actions = (best_actions * S[:, None, None])
        mean_act = np.sum(weighted_actions, 0) / (np.sum(S) + 1e-4)

        cov = None
        if self._hp.refit_cov:
            cov = np.cov(np.transpose(best_actions.reshape(best_actions.shape[0], -1)))
        
        return self._sample_noise(n_samples, cov) + mean_act.reshape((1, best_actions.shape[1], self._adim))

    @staticmethod
    def get_default_hparams():
        hparams_dict = {
            'nactions': 15,
            'initial_std': [0.05, 0.05, 0.2, np.pi / 10],
            'mean_bias': np.zeros(4),
            'kappa': 1,
            'beta_0': 0.5,
            'beta_1': 0.5,
            'smooth_across_last_action': False,
            'refit_cov': False
        }
        return hparams_dict
