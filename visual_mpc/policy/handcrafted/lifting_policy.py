from ..policy import Policy
import numpy as np


class LiftingPolicy(Policy):
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)

        if self._hp.action_space == 'xzgrasp':
            assert self._hp.nactions >= 5, "Need at least 5 actions"
            assert all([x > 0 for x in self._hp.frac_act]) and sum(self._hp.frac_act) <= 1.
            assert ag_params['adim'] == 3, "xzgrasp should have adim = 3"
        else:
            raise NotImplementedError

    def _default_hparams(self):
        default_dict = {
            'nactions': 15,
            'repeat': 1,
            'action_space': 'xzgrasp',
            'frac_act': [0.4, 0.1],
            'sigma': [0.05, 0.1, 0],
            'bounds': [[-0.4, 0.05], [0.4, 0.15]],
            'up_z': 0.15,
            'floor_z': -0.075
        }

        parent_params = super(LiftingPolicy, self)._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def act(self, t, state, object_poses):
        if self._hp.action_space == 'xzgrasp':
            return self._act_xzgrasp(t, state, object_poses)
        raise NotImplementedError

    def reset(self):
        if self._hp.action_space == 'xzgrasp':
            self._actions = None
            return
        raise NotImplementedError

    def _act_xzgrasp(self, t, state, object_poses):
        if t == 0:
            target_pos = np.random.uniform(low=self._hp.bounds[0], high=self._hp.bounds[1])
            n_movement_actions = self._hp.nactions - 1
            actions = np.zeros((self._hp.nactions, 3))
            chosen_ind = np.random.choice(object_poses.shape[1])
            t_move_1, t_down = [int(max(np.round(n_movement_actions * x), 1))
                                                for x in self._hp.frac_act]
            t_move_2 = n_movement_actions - t_move_1 - t_down
            assert t_move_2 > 0, "Not enough time to move object"

            delta_x1 = object_poses[0, chosen_ind, 0] - state[0, 0]
            actions[:t_move_1] = [delta_x1 / t_move_1, (self._hp.up_z - state[0, 1])/ t_move_1, -1]

            actions[t_move_1: (t_down + t_move_1)] = [0, (self._hp.floor_z - self._hp.up_z) / t_down, -1]
            actions[t_down + t_move_1] = [0, 0, 1]

            delta_x2 = target_pos[0] - object_poses[0, chosen_ind, 0]
            actions[t_down + t_move_1 + 1:] = [delta_x2 / t_move_2, (target_pos[1] - self._hp.floor_z) / t_move_2, 1]

            actions += np.random.normal(size=(self._hp.nactions, 3)) * self._hp.sigma
     
            actions = np.repeat(actions, self._hp.repeat, axis=0)
            actions[:,:2] /= self._hp.repeat

            self._actions = actions

        return {'actions': self._actions[t].copy()}
