"""
Implements a custom sampler for the CEM controller
"""

class CEMSampler(object):
    def __init__(self, hp, adim, sdim, **kwargs):
        self._hp = hp
        self._adim, self._sdim = adim, sdim
        self._chosen_actions = []

    def sample_initial_actions(self, nsamples, current_state):
        """
        Samples initial actions for CEM iterations
        :param nsamples: number of samples
        :param current_state: Current state of robot
        :return: action samples in (B, T, adim) format
        """

        raise NotImplementedError

    def sample_next_actions(self, n_samples, best_actions):
        """
        Samples actions for CEM iterations, given BEST last actions
        :param nsamples: number of samples
        :param best_actions: number of samples
        :return: action samples in (B, T, adim) format
        """
        raise NotImplementedError

    def log_best_action(self, action):
        """
        Some sampling distributions may change given the taken action

        :param action: action executed
        :return: None
        """
        self._chosen_actions.append(action.copy())

    @staticmethod
    def get_default_hparams():
        hparams_dict = {
        }
        return hparams_dict
