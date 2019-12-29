import numpy as np
from visual_mpc.policy.utils.controller_utils import truncate_movement
from .cem_sampler import CEMSampler

class AutograspEpsilon(CEMSampler):
    def __init__(self, sigma, mean, hp, repeat, base_adim):
        super(AutograspEpsilon, self).__init__(sigma, mean, hp, repeat, base_adim)

        self._hp = hp
        assert 0 <= self._hp.base_frac <= 1, "base_frac should be a valid probability"
        assert 0 <= self._hp.base_frac_reduce < 1, "base_frac_reduce should be in [0, 1)"
        assert 0 <= self._hp.ag_epsilon <= 1, "epsilon should be a valid probability"

        z_dim, gripper_dim = 2, -1
        if self._hp.action_order[0] is not None:
            assert 'z' in hp.action_order and 'grasp' in hp.action_order, "Ap epsilon requires z and grasp action"
            for i, a in enumerate(hp.action_order):
                if a == 'grasp':
                    gripper_dim = i
                elif a == 'z':
                    z_dim = i

        naction_steps = hp.nactions
        self._adim = adim
        self._repeat = repeat
        self._steps = naction_steps
        self._z_dim, self._gripper_dim = z_dim, gripper_dim

    def _default_sampler(self, mean, sigma, M):
        actions = np.random.multivariate_normal(mean, sigma, M)
        actions = actions.reshape(M, self._hp.nactions, self._adim)
        if self._hp.action_bound:
            actions = truncate_movement(actions, self._hp)

        actions = np.repeat(actions, self._hp.repeat, axis=1)
        return actions

    def _apply_ag_epsilon(self, state, actions, close_override):
        cumulative_zs = np.cumsum(actions[:, :, self._z_dim] / self._hp.z_norm, 1) + state[-1, self._z_dim]
        z_thresh_check = (cumulative_zs <= self._hp.z_thresh).astype(np.float32) * 2 - 1
        first_close_pos = np.argmax(z_thresh_check, axis=1)
        if close_override:
            actions[:, :, self._gripper_dim] = 1
        else:
            for i, p in enumerate(first_close_pos):
                pivot = p - p % self._hp.repeat  # ensure that pivots only occur on repeat boundry
                actions[i, :pivot, self._gripper_dim] = -1
                actions[i, pivot:, self._gripper_dim] = 1
        epsilon_vec = np.random.choice([-1, 1], size=actions.shape[:-1], p=[self._hp.ag_epsilon, 1 - self._hp.ag_epsilon])
        actions[:, :, self._gripper_dim] *= epsilon_vec

    def sample(self, itr, M, current_state, current_mean, current_sigma, close_override):
        apply_amount = max(int(M * self._hp.base_frac * (self._hp.base_frac_reduce ** itr)), 1)

        actions = self._default_sampler(current_mean, current_sigma, M)
        self._apply_ag_epsilon(current_state, actions[:apply_amount], close_override)
        return actions

    @staticmethod
    def get_default_hparams():
        hparams_dict = {
            'z_thresh': [1. / 5, 1. / 5, 1. / 3],
            'ag_epsilon': 0.5,
            'z_norm': 1,
            'base_frac': 1,
            'base_frac_reduce': 0.3
        }
        return hparams_dict
