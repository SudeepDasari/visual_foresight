import numpy as np
from visual_mpc.utils.logger import Logger
from visual_mpc.policy.policy import Policy
from robonet.inverse_model.testing.action_inference_interface import ActionInferenceInterface

class InvModelBaseController(Policy):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        """
        :param ag_params:
        :param policyparams:
        """
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)

        self.agentparams = ag_params

        if self._hp.logging_dir:
            self._logger = Logger(self._hp.logging_dir, 'cem{}log.txt'.format(self.agentparams['gpu_id']))
        else:
            self._logger = Logger(printout=True)

        self._logger.log('init inverse model controller')

        #action dimensions:
        self._adim = self.agentparams['adim']
        self._sdim = self.agentparams['sdim']                             # state dimension

        predictor_hparams = {'load_T':self._hp.load_T}
        self.predictor = self._hp.predictor_class(self._hp.model_params_path, predictor_hparams, n_gpus=ngpu, first_gpu=gpu_id)
        self.predictor.restore(self._hp.model_restore_path)

        self.action_counter = 0
        self.actions = None

    def _default_hparams(self):
        default_dict = {
            'T': 15,                       # planning horizon
            'predictor_class': ActionInferenceInterface,
            'model_params_path': '',
            'model_restore_path': '',
            'logging_dir':'',
            'load_T':7
        }

        parent_params = super(InvModelBaseController, self)._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def reset(self):
        self.plan_stat = {} #planning statistics

    def act(self, t=None, i_tr=None, images=None, goal_image=None):
        if t % (self._hp.load_T - 1) == 0:
            self.actions = self.predictor(convert_to_float(images[-1,0]), goal_image[-1, 0])  # select last-image and 0-th camera
            self.action_counter = 0
        print('t {} action counter {}'.format(t, self.action_counter))
        action = self.actions[0, self.action_counter]
        self.action_counter += 1
        print('action ', action)
        return {'actions': action, 'plan_stat':self.plan_stat}

def convert_to_float(input):
    assert input.dtype == np.uint8, "assumed input is uint8"
    return input.astype(np.uint8) / 255.



