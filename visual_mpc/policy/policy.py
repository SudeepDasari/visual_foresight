""" This file defines the base class for the policy. """
import abc, six
from funcsigs import signature, Parameter
from tensorflow.contrib.training import HParams
import numpy as np
import pdb


def get_policy_args(policy, obs, t, i_tr, step_data=None):
    """
    Generalized way to get data from agent and pass to policy
    :param policy: Subclass of Policy
    :param obs: obs_dict from agent
    :param t: current timestep
    :param i_tr: current traj number
    :param step_data: dict of step specific data passed in by agent
    :return: dictionary mapping keys that the policy requests (by way of argument in policy.act) to their value
    """

    policy_args = {}
    policy_signature = signature(policy.act)  # Gets arguments required by policy
    for arg in policy_signature.parameters:  # Fills out arguments according to their keyword
        value = policy_signature.parameters[arg].default
        if arg in obs:
            value = obs[arg]
        elif step_data is not None and arg in step_data:
            value = step_data[arg]

        # everthing that is not cached in post_process_obs is assigned here:
        elif arg == 't':
            value = t
        elif arg == 'i_tr':
            value = i_tr
        elif arg == 'obs':           # policy can ask for all arguments from environment
            value = obs
        elif arg == 'step_data':
            value = step_data
        elif arg == 'goal_pos':
            value = step_data['goal_pos']

        if value is Parameter.empty:
            # required parameters MUST be set by agent
            raise ValueError("Required Policy Param {} not set in agent".format(arg))
        policy_args[arg] = value
    # import pdb; pdb.set_trace()
    return policy_args


@six.add_metaclass(abc.ABCMeta)
class Policy(object):
    def _override_defaults(self, policyparams):
        for name, value in policyparams.items():
            if name == 'type':
                continue      # type corresponds to policy class

            print('overriding param {} to value {}'.format(name, value))
            if np.all(value == getattr(self._hp, name)):
                raise ValueError("attribute is {} is identical to default value!!".format(name))

            if name in self._hp and self._hp.get(name) is None:   # don't do a type check for None default values
                setattr(self._hp, name, value)
            else:
                self._hp.set_hparam(name, value)

    def _default_hparams(self):
        return HParams()

    @abc.abstractmethod
    def act(self, *args):
        """
        Args:
            Request necessary arguments in definition
            (see Agent code)
        Returns:
            A dict of outputs D
               -One key in D, 'actions' should have the action for this time-step
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        pass


class DummyPolicy(object):
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        """ Computes actions from states/observations. """
        pass

    @abc.abstractmethod
    def act(self, *args):
        pass

    def reset(self):
        pass


class NullPolicy(Policy):
    """
    Returns 0 for all timesteps
    """
    def __init__(self,  ag_params, policyparams, gpu_id, ngpu):
        self._adim = ag_params['adim']
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)

    def _default_hparams(self):
        default_dict = {
            'wait_for_user': False
        }
        parent_params = super(NullPolicy, self)._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def act(self):
        if self._hp.wait_for_user:
            pdb.set_trace()
        return {'actions': np.zeros(self._adim)}