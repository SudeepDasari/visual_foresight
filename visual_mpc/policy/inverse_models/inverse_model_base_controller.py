import numpy as np
from visual_mpc.utils.logger import Logger
from visual_mpc.policy.policy import Policy
from robonet.inverse_model.testing.action_inference_interface import ActionInferenceInterface
from .samplers import GaussianCEMSampler

class InvModelBaseController(Policy):
    """
    Inverse model policy
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
        self._sdim = self.agentparams['sdim']
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
            'replan_every': 2,
            'context_action_weight': [0.5, 0.5, 0.05, 1],
            'sampler': GaussianCEMSampler,
        }

        parent_params = super(InvModelBaseController, self)._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def reset(self):
        self.plan_stat = {} #planning statistics
        self.action_counter = 0
        self.actions = None
        self.context_actions = [None] * self._hp.num_context
        self.context_frames = [None] * self._hp.num_context

    def act(self, t=None, i_tr=None, images=None, goal_image=None):

        if t < self._hp.num_context:
            initial_sampler = self._hp.sampler(self._hp, self._adim, self._sdim)
            action = initial_sampler.sample_initial_actions(t, 1, state[-1])[0, 0] * self._hp.context_action_weight

        elif t >= self._hp.num_context:
            if ((t - self._hp.num_context) % self._hp.replan_every == 0):
                float_ctx = [frame[None, None] for frame in self.context_frames] 
                prepped_ctx_im = np.concatenate(float_ctx, axis=1)
                prepped_ctx_act = np.array(self.context_actions)[None] 
                self.actions = self.predictor(convert_to_float(images[-1,0]), goal_image[-1, 0],
                                              prepped_ctx_act, prepped_ctx_im)  # select last-image and 0-th camera
                self.action_counter = 0
            print('t {} action counter {}'.format(t, self.action_counter))
            action = self.actions[0, self.action_counter]
            self.action_counter += 1
        else:
            action = np.zeros(self._adim)
       
        print('action ', action)
        new_context_image = convert_to_float(np.copy(images[-1, 0]))
        self.update_context(new_context_image, action)
        return {'actions': action, 'plan_stat':self.plan_stat}

    def update_context(self, new_image, new_action):
        self.context_frames.append(new_image)
        self.context_actions.append(new_action)
        if len(self.context_frames) > self._hp.num_context:
            self.context_frames.pop(0)
            self.context_actions.pop(0)
        
def convert_to_float(input):
    assert input.dtype == np.uint8, "assumed input is uint8"
    return input.astype(np.uint8) / 255.

