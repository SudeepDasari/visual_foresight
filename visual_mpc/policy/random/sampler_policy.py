import pdb
import numpy as np
from visual_mpc.policy.policy import Policy
from visual_mpc.policy.cem_controllers.samplers.correlated_noise import CorrelatedNoiseSampler

class SamplerPolicy(Policy):
    """
    Random Policy
    """

    def __init__(self, agentparams, policyparams, gpu_id, npgu, **kwargs):

        self._hp = self._default_hparams()
        self.override_defaults(policyparams)
        self.agentparams = agentparams
        self.adim = agentparams.adim

        self._sampler = self._hp.sampler(policyparams, self.adim)
        self._actions = None

    def _default_hparams(self):
        default_dict = {
            'nactions': 15,
            'sampler':CorrelatedNoiseSampler,
            'initial_std':  [0.05, 0.05, 0.2, np.pi / 10],
            'beta_0': 0.5,
            'beta_1': 0.5,
        }
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def act(self, t):
        if t == 0:
            self._actions = self._sampler.sample_initial_actions(n_samples=1).squeeze()
        return {'actions': self._actions[t]}


