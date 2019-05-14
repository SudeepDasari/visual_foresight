""" This file defines the linear Gaussian policy class. """
import numpy as np
from visual_mpc.policy.policy import Policy
from visual_mpc.policy.utils.controller_utils import construct_initial_sigma, truncate_movement
from visual_mpc.envs.util.action_util import autograsp_grip_logic


class GaussianPolicy(Policy):
    """
    Random Policy
    """

    def __init__(self, agentparams, policyparams, gpu_id, npgu):

        self._hp = self._default_hparams()
        self._override_defaults(policyparams)
        self.agentparams = agentparams
        self.adim = agentparams['adim']

    def _default_hparams(self):
        default_dict = {
            'nactions': 5,
            'repeat': 3,
            'action_bound': True,
            'action_order': None,
            'initial_std': 0.05,  # std dev. in xy
            'initial_std_lift': 0.15,  # std dev. in xy
            'initial_std_rot': np.pi / 18,
            'initial_std_grasp': 2.,
            'type': None,
            'discrete_gripper': None
        }

        parent_params = super(GaussianPolicy, self)._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def act(self, t):
        assert self.agentparams['T'] == self._hp.nactions * self._hp.repeat
        if t == 0:
            mean = np.zeros(self.adim * self._hp.nactions)
            # initialize mean and variance of the discrete actions to their mean and variance used during data collection
            sigma = construct_initial_sigma(self._hp, self.adim)
            self.actions = np.random.multivariate_normal(mean, sigma).reshape(self._hp.nactions, -1)
            self.process_actions()
        return {'actions': self.actions[t, :self.adim]}

    def process_actions(self):
        if len(self.actions.shape) == 2:
            self.actions = self._process(self.actions)
        elif len(self.actions.shape) == 3:  # when processing batch of actions
            newactions = []
            for b in range(self.actions.shape[0]):
                newactions.append(self._process(self.actions[b]))
            self.actions = np.stack(newactions, axis=0)

    def _process(self, actions):
        if self._hp.discrete_gripper:
            actions = discretize_gripper(actions, self._hp.discrete_gripper)
        if self._hp.action_bound:
            actions = truncate_movement(actions, self._hp)

        actions = np.repeat(actions, self._hp.repeat, axis=0)
        return actions

    def finish(self):
        pass


def discretize_gripper(actions, gripper_ind):
    assert len(actions.shape) == 2
    for a in range(actions.shape[0]):
        if actions[a, gripper_ind] >= 0:
            actions[a, gripper_ind] = 1
        else:
            actions[a, gripper_ind] = -1
    return actions


class GaussianAGEpsilonPolicy(GaussianPolicy):
    def _default_hparams(self):
        default_dict = {
            'p_epsilon': 0.15,
            'zthresh': 0.15,
            'gripper_joint_thresh': -1.,
            'reopen': True,
            'grip_cmds': [1.0, -1.0]
        }

        parent_params = super(GaussianAGEpsilonPolicy, self)._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params
    
    def act(self, t, state, finger_sensors):
        parent_action = super(GaussianAGEpsilonPolicy, self).act(t)['actions']

        if t == 0:
            self._last_grip = None
            self._prev_touch = False

        if t % self._hp.repeat == 0:
            joint_test = state[-1, -1] > 0 and np.abs(state[-1, -1]) < self._hp.gripper_joint_thresh
            touch_test = joint_test or np.amax(finger_sensors[-1]) > 0

            self._last_grip = autograsp_grip_logic(state[-1, 2], self._hp.zthresh, self._last_grip, 
                                                   self._hp.reopen, touch_test or self._prev_touch)
            self._prev_touch = touch_test
        
        bool_cast = lambda x: self._hp.grip_cmds[0] if x else self._hp.grip_cmds[1]
        if np.random.uniform() < self._hp.p_epsilon:
            print('flip')
            grip_cmd = bool_cast(not self._last_grip)
        else:
            grip_cmd = bool_cast(self._last_grip)
        
        parent_action[-1] = grip_cmd
        return {'actions': parent_action}
