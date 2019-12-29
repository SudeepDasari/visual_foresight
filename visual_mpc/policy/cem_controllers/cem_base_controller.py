import numpy as np
from visual_mpc.utils.logger import Logger
from .samplers import GaussianCEMSampler
from visual_mpc.policy.policy import Policy


class CEMBaseController(Policy):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams):
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

        self._logger.log('init CEM controller')

        self._t_since_replan = None
        self._t = None
        self._n_iter = self._hp.iterations

        #action dimensions:
        self._adim = self.agentparams['adim']
        self._sdim = self.agentparams['sdim']                             # state dimension

        self._sampler = None
        self._best_indices, self._best_actions = None, None

        self._state = None
        assert self._hp.minimum_selection > 0, "must take at least 1 sample for refitting"

    def _default_hparams(self):
        default_dict = {
            'append_action': None,
            'verbose': True,
            'verbose_every_iter': False,
            'logging_dir': '',
            'hard_coded_start_action': None,
            'context_action_weight': [0.5, 0.5, 0.05, 1],
            'zeros_for_start_frames': True,
            'replan_interval': 0,
            'sampler': GaussianCEMSampler,
            'T': 15,                       # planning horizon
            'iterations': 3,
            'num_samples': 200,
            'selection_frac': 0., # specifcy which fraction of best samples to use to compute mean and var for next CEM iteration
            'start_planning': 0,
            'minimum_selection': 10
        }

        parent_params = super(CEMBaseController, self)._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def _override_defaults(self, policyparams):
        sampler_class = policyparams.get('sampler', GaussianCEMSampler)
        for name, value in sampler_class.get_default_hparams().items():
            if name in self._hp:
                print('Warning default value for {} already set!'.format(name))
                self._hp.set_hparam(name, value)
            else:
                self._hp.add_hparam(name, value)

        super(CEMBaseController, self)._override_defaults(policyparams)
        self._hp.sampler = sampler_class

    def reset(self):
        self._best_indices = None
        self._best_actions = None
        self._t_since_replan = None
        self._sampler = self._hp.sampler(self._hp, self._adim, self._sdim)
        self.plan_stat = {} #planning statistics

    def perform_CEM(self, state):
        self._logger.log('starting cem at t{}...'.format(self._t))
        self._logger.log('------------------------------------------------')

        K = self._hp.minimum_selection
        if self._hp.selection_frac:
            K = max(int(self._hp.selection_frac * self._hp.num_samples), self._hp.minimum_selection)
        actions = self._sampler.sample_initial_actions(self._t, self._hp.num_samples, state[-1])
        for itr in range(self._n_iter):
            if self._hp.append_action:
                act_append = np.tile(np.array(self._hp.append_action)[None, None], [self._hp.num_samples, actions.shape[1], 1])
                actions = np.concatenate((actions, act_append), axis=-1)
            
            self._logger.log('------------')
            self._logger.log('iteration: ', itr)

            scores = self.evaluate_rollouts(actions, itr)
            assert scores.shape == (actions.shape[0],), "score shape should be (n_actions,)"

            self._best_indices = scores.argsort()[:K]
            self._best_actions = actions[self._best_indices]

            self.plan_stat['scores_itr{}'.format(itr)] = scores
            if itr < self._n_iter - 1:
                re_sample_act = self._best_actions.copy()
                if self._hp.append_action:
                    re_sample_act = re_sample_act[:, :, :-len(self._hp.append_action)]
                
                actions = self._sampler.sample_next_actions(self._hp.num_samples, re_sample_act, scores[self._best_indices].copy())

      
        self._t_since_replan = 0

    def evaluate_rollouts(self, actions, cem_itr):
        raise NotImplementedError

    def _verbose_condition(self, cem_itr):
        if self._hp.verbose:
            if self._hp.verbose_every_iter or cem_itr == self._n_iter - 1:
                return True
        return False

    def act(self, t=None, i_tr=None, state=None):
        """
        Return a random action for a state.
        Args:
            t: the current controller's Time step
        """
        self._state = state
        self.i_tr = i_tr
        self._t = t

        if t < self._hp.start_planning:
            if self._hp.zeros_for_start_frames:
                assert self._hp.hard_coded_start_action is None
                action = np.zeros(self.agentparams['adim'])
            elif self._hp.hard_coded_start_action:
                action = np.array(self._hp.hard_coded_start_action)
            else:
                initial_sampler = self._hp.sampler(self._hp, self._adim, self._sdim)
                action = initial_sampler.sample_initial_actions(t, 1, state[-1])[0, 0] * self._hp.context_action_weight
                if self._hp.append_action:
                    action = np.concatenate((action, self._hp.append_action), axis=0)
                
        else:
            if self._hp.replan_interval:
                if self._t_since_replan is None or self._t_since_replan + 1 >= self._hp.replan_interval:
                    self.perform_CEM(state)
                else:
                    self._t_since_replan += 1
            else:
                self.perform_CEM(state)
            action = self._best_actions[0, self._t_since_replan]

        assert action.shape == (self.agentparams['adim'],), "action shape does not match adim!"

        self._logger.log('time {}, action - {}'.format(t, action))
        
        if self._best_actions is not None:
            action_plan_slice = self._best_actions[:, min(self._t_since_replan + 1, self._hp.T - 1):]
            self._sampler.log_best_action(action, action_plan_slice)
        else:
            self._sampler.log_best_action(action, None)

        return {'actions':action, 'plan_stat':self.plan_stat}
