from visual_mpc.policy.policy import Policy
import numpy as np
from visual_mpc.policy.utils.controller_utils import truncate_movement
import copy


def round_up(val, round):
    return val + (-val % round)


class RandomFoldPolicy(Policy):
    def __init__(self, agent_params, policyparams, gpu_id, ngpu):
        assert agent_params['adim'] == 4, "Action dimension should be 4 for this policy!"
        self._adim, self._T = agent_params['adim'], agent_params['T']
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)
        self.agent_params = agent_params
        self._swap_times, self._stage, self._ctr = [], 0, 0
        self._last_action = None
        self._pick_point, self._drop_point = None, None

    def _default_hparams(self):
        default_dict = {
            'repeat': 3,
            'action_bound': False,
            'action_order': [None],
            'switch_prob': 0.25,
            'initial_std': 0.005,
            'initial_std_lift': 0.05,  # std dev. in xy
            'initial_std_rot': np.pi / 18,
            'max_z_shift': 1./3,
            'min_dist': 0.8,
            'pick_timer': 3            # time for go down/pick actions
        }

        parent_params = super(RandomFoldPolicy, self)._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def _override_defaults(self, policyparams):
        assert policyparams.get('repeat', 3) >= 1, "Repeat must be at least 1"
        return super(RandomFoldPolicy, self)._override_defaults(policyparams)

    def _is_timer_set(self):
        return self._ctr > 0

    def _tick(self, ret_val):
        self._ctr -= 1
        if self._ctr == 0:
            self._stage += 1
        if self._hp.action_bound:
            ret_val['actions'] = truncate_movement(ret_val['actions'][None], self._hp)[0]
        return ret_val

    def _set_timer(self, countdown):
        self._ctr = countdown

    def act(self, t, state):
        if t == 0:
            action_time = round_up(self._hp.pick_timer, self._hp.repeat)
            move_time1 = self._T + 1
            while move_time1 > self._T - 3 * action_time - self._hp.repeat:
                move_time1 = round_up(np.random.geometric(self._hp.switch_prob), self._hp.repeat)
            move_time2 = self._T - 3 * action_time - move_time1

            pick_point, drop_point = np.zeros(2), np.zeros(2)
            while np.linalg.norm(pick_point - drop_point) < self._hp.min_dist:
                pick_point, drop_point = np.random.uniform(size=2),np.random.uniform(size=2)
            self._pick_point, self._drop_point = pick_point, drop_point
            self._swap_times = [move_time1, action_time, action_time, move_time2,
                                action_time]
            self._stage, self._ctr = 0, 0

        if not self._is_timer_set():
            self._set_timer(self._swap_times[self._stage])

        if self._stage == 0 or self._stage == 3:
            print(t, 'rand move: {}'.format(self._stage))
            if t % self._hp.repeat == 0:
                xyz_std, rot_std = self._hp.initial_std, self._hp.initial_std_rot
                mean = np.zeros(self._adim)
                dest = self._pick_point
                if self._stage > 0:
                    dest = self._drop_point
                    rot_std /= 5.
                mean[0:2] = (dest - state[-1, :2]) / self._ctr
                if state[-1, 2] < 0.5:
                    mean[2] = 1     # bias actions to move gripper upwards and avoid towel
                elif self._stage > 0:
                    mean[2] = 0.1

                sigma = np.diag([xyz_std, xyz_std, self._hp.initial_std_lift, rot_std])
                self._last_action = np.random.multivariate_normal(mean, sigma)

                if self._hp.max_z_shift > 0:
                    self._last_action[2] = np.clip(self._last_action[2], -self._hp.max_z_shift, self._hp.max_z_shift)
            return self._tick({'actions': copy.deepcopy(self._last_action)})

        elif self._stage == 1 or self._stage == 4:
            print(t, 'down')
            if t % self._hp.repeat == 0:
                # pdb.set_trace()
                mean = np.array([0., 0., -1, 0])
                sigma = np.diag([self._hp.initial_std / 5., self._hp.initial_std / 5.,
                                 self._hp.initial_std_lift / 2., self._hp.initial_std_rot / 10.])
                self._last_action = np.random.multivariate_normal(mean, sigma)
                if self._hp.max_z_shift > 0:
                    self._last_action[2] = np.clip(self._last_action[2], -self._hp.max_z_shift, self._hp.max_z_shift)
            return self._tick({'actions': copy.deepcopy(self._last_action)})

        elif self._stage == 2:
            print(t, 'up')
            if t % self._hp.repeat == 0:
                # pdb.set_trace()
                mean = np.array([0., 0., 1, 0])
                sigma = np.diag([self._hp.initial_std / 10., self._hp.initial_std / 10.,
                                 self._hp.initial_std_lift / 2., self._hp.initial_std_rot / 10.])
                self._last_action = np.random.multivariate_normal(mean, sigma)
                if self._hp.max_z_shift > 0:
                    self._last_action[2] = np.clip(self._last_action[2], -self._hp.max_z_shift, self._hp.max_z_shift)
            return self._tick({'actions': copy.deepcopy(self._last_action)})

        else:
            raise ValueError("stage {} not defined".format(self._stage))
