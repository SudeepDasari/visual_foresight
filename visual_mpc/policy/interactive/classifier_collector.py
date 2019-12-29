import numpy as np
from visual_mpc.policy.policy import Policy


class CollectExamplesPolicy(Policy):
    def __init__(self, agentparams, policyparams, gpu_id, npgu):
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)
        self.agentparams = agentparams
        self._adim = agentparams['adim']
        assert self._adim == 5, "only adim=5 supported"

    def _default_hparams(self):
        parent_params = super(CollectExamplesPolicy, self)._default_hparams()
        parent_params.add_hparam('floor', [0., 0., 0.1, 0.])
        parent_params.add_hparam('ceil', [1., 1., 1., 0.])
        parent_params.add_hparam('gripper_prob', 0.5)
        return parent_params

    def act(self, state, t):
        next_act = np.zeros(self._adim)
        next_act[:4] = np.random.uniform(self._hp.floor, self._hp.ceil) - state[-1, :4]
        if np.random.uniform() <= self._hp.gripper_prob:
            next_act[-1] = 1
        else:
            next_act[-1] = -1
        return {'actions': next_act}
