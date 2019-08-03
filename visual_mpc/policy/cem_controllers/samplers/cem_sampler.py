"""
Implements a custom sampler for the CEM controller
"""
import numpy as np


class CEMSampler(object):
    def __init__(self, hp, adim, sdim, **kwargs):
        self._hp = hp
        self._adim, self.b_sdim = adim, sdim
        self._chosen_actions = []
        self._best_action_plans = []

    def sample_initial_actions(self, t, nsamples, current_state):
        """
        Samples initial actions for CEM iterations
        :param nsamples: number of samples
        :param current_state: Current state of robot
        :return: action samples in (B, T, adim) format
        """

        raise NotImplementedError

    def sample_next_actions(self, n_samples, best_actions, scores):
        """
        Samples actions for CEM iterations, given BEST last actions
        :param nsamples: number of samples
        :param best_actions: number of samples
        :return: action samples in (B, T, adim) format
        """
        raise NotImplementedError

    def log_best_action(self, action, best_action_plans):
        """
        Some sampling distributions may change given the taken action

        :param action: action executed
        :param best_action_plans: batch of next planned actions (after this timestep) ordered in ascending cost
        :return: None
        """
        self._chosen_actions.append(action.copy())
        self._best_action_plans.append(best_action_plans)
    
    @property
    def chosen_actions(self):
        """
        :return: actions chosen by policy thus far
        """
        return np.array(self._chosen_actions)

    @staticmethod
    def get_default_hparams():
        hparams_dict = {
        }
        return hparams_dict
