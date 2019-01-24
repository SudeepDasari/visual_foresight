from visual_mpc.policy.cem_controllers import CEMControllerBase
import imp
import control_embedding
import numpy as np


class NCECostController(CEMControllerBase):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        """

        :param ag_params: agent parameters
        :param policyparams: policy parameters
        :param gpu_id: starting gpu id
        :param ngpu: number of gpus
        """
        CEMControllerBase.__init__(self, ag_params, policyparams)

        params = imp.load_source('params', ag_params['current_dir'] + '/conf.py')
        net_conf = params.configuration

        if ngpu > 1:
            vpred_ngpu = ngpu - 1
        else: vpred_ngpu = ngpu

        self._predictor = net_conf['setup_predictor'](ag_params, net_conf, gpu_id, vpred_ngpu, self._logger)
        self._scoring_func = control_embedding.deploy_model(self._hp.nce_conf_path, batch_size=self._hp.nce_batch_size,
                                                            restore_path=self._hp.nce_restore_path,
                                                            device_id=gpu_id + ngpu - 1)

        self._b_size = net_conf['batch_size']
        self._seqlen = net_conf['sequence_length']

        self._n_contxt = net_conf['context_frames']

        self._img_height, self._img_width = net_conf['orig_size']

        self._n_cam = net_conf['ncam']

        self._images = None
        self._goal_image = None
        self._start_image = None

    def _default_hparams(self):
        default_dict = {
            'nce_conf_path': '',
            'nce_restore_path': '',
            'nce_batch_size': 200,
            'state_append': None
        }
        parent_params = super(NCECostController, self)._default_hparams()

        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def evaluate_rollouts(self, actions, cem_itr):
        raise NotImplementedError

    def _prep_vidpred_inp(self, actions):
        ctxt = self._n_contxt
        last_frames = self._images[self._t - ctxt + 1:self._t + 1]  # same as [t - 1:t + 1] for context 2
        last_frames = last_frames.astype(np.float32, copy=False) / 255.
        last_frames = last_frames[None]
        last_states = self.state[self._t - ctxt + 1:self._t + 1]
        last_states = last_states[None]
        if self._hp.state_append:
            last_state_append = np.tile(np.array([[self._hp.state_append]]), (1, ctxt, 1))
            last_states = np.concatenate((last_states, last_state_append), -1)

        return actions, last_frames, last_states

    def act(self, t=None, i_tr=None, goal_image=None, images=None, state=None):
        """
        Return a random action for a state.
        Args:
            if performing highres tracking images is highres image
            t: the current controller's Time step
            goal_pix: in coordinates of small image
            desig_pix: in coordinates of small image
        """
        self._start_image = images[-1]
        self._goal_image = goal_image[1]
        self._images = images

        return super(NCECostController, self).act(t, i_tr)
