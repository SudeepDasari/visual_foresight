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

        #predictor_hparams = {'load_T':self._hp.load_T}
	predictor_hparams = {}
        self.predictor = self._hp.predictor_class(self._hp.model_params_path, predictor_hparams, n_gpus=ngpu, first_gpu=gpu_id)
        self.predictor.restore()

        self.action_counter = 0
        self.actions = None
        self.context_actions = [None] * self._hp.num_context
        self.context_frames = [None] * self._hp.num_context

    def _default_hparams(self):
        default_dict = {
            'T': 15,                       # planning horizon
            'predictor_class': ActionInferenceInterface,
            'model_params_path': '',
            'model_restore_path': '',
            'logging_dir':'',
            'load_T':7,
            'num_context': 2,
        }

        parent_params = super(InvModelBaseController, self)._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def reset(self):
        self.plan_stat = {} #planning statistics

    def act(self, t=None, i_tr=None, images=None, goal_image=None):
        #if t % (self._hp.load_T - 1) == 0:
        if t < self._hp.num_context:
            self.context_frames[t] = convert_to_float(np.copy(images[-1, 0]))
            action = np.array([
                        np.random.uniform(-0.025, 0.025),
                        np.random.uniform(-0.025, 0.025),
                        np.random.uniform(-0.025, 0.025),
                        0
                    ])
            self.context_actions[t] = action
        elif t >= self._hp.num_context:
            float_ctx = [frame[None, None] for frame in self.context_frames] 
            prepped_ctx = np.concatenate(float_ctx, axis=1)
            self.actions = self.predictor(convert_to_float(images[-1,0]), goal_image[-1, 0],
                                          np.array(self.context_actions)[None], prepped_ctx)  # select last-image and 0-th camera
            self.action_counter = 0
            print('t {} action counter {}'.format(t, self.action_counter))
            action = self.actions[0, self.action_counter]
            self.action_counter += 1
            self.context_frames.append(convert_to_float(np.copy(images[-1, 0])))
            self.context_frames.pop()
            self.context_actions.append(action)
            self.context_actions.pop()
        else:
            action = np.zeros(4)
        print('action ', action)
        return {'actions': action, 'plan_stat':self.plan_stat}

def convert_to_float(input):
    assert input.dtype == np.uint8, "assumed input is uint8"
    return input.astype(np.uint8) / 255.



