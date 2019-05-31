import numpy as np
from visual_mpc.policy.policy import Policy
import sys
if sys.version_info[0] < 3:
    import cPickle as pkl
else:
    import pickle as pkl


class PlaybackPolicy(Policy):
    def __init__(self, agentparams, policyparams, gpu_id, npgu):
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)
        self.agentparams = agentparams
        self._adim = agentparams['adim']
        self._pkl = None

    def _default_hparams(self):
        parent_params = super(PlaybackPolicy, self)._default_hparams()
        parent_params.add_hparam('file', './act.pkl')
        return parent_params

    def act(self, state, t):
        if t == 0 or self._pkl is None:
            self._pkl = pkl.load(open(self._hp.file, 'rb'))
        assert 0 <= t < len(self._pkl), "too long!"

        return {'actions': self._pkl[t]['actions']}
