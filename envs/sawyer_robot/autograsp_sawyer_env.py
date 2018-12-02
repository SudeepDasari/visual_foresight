from envs.sawyer_robot.base_sawyer_env import BaseSawyerEnv
import copy
from envs.util.action_util import autograsp_dynamics
import numpy as np

class AutograspSawyerEnv(BaseSawyerEnv):
    def __init__(self, env_params, _=None):
        assert 'mode_rel' not in env_params, "Autograsp sets mode_rel"

        self._hyper = copy.deepcopy(env_params)

        BaseSawyerEnv.__init__(self, self._hyper)
        self._adim, self._sdim = 4, self._base_sdim

    def _init_dynamics(self):
        self._gripper_closed = False
        self._prev_touch = False

    def _default_hparams(self):
        default_dict = {'zthresh': 0.15,
                        'gripper_joint_thresh': -1.,   # anything <0 deactivates this check
                        'reopen': True}

        parent_params = BaseSawyerEnv._default_hparams(self)
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def _next_qpos(self, action):
        assert action.shape[0] == 4      # z dimensions are normalized across robots
        norm_gripper_z = (self._previous_target_qpos[2] - self._low_bound[2]) / \
                         (self._high_bound[2] - self._low_bound[2])
        z_thresh = self._hp.zthresh

        joint_test = self._last_obs['state'][-1] > 0 and \
                     np.abs(self._last_obs['state'][-1]) < self._hp.gripper_joint_thresh
        touch_test = joint_test or np.amax(self._last_obs['finger_sensors']) > 0

        target, self._gripper_closed = autograsp_dynamics(self._previous_target_qpos, action,
                                                          self._gripper_closed, norm_gripper_z, z_thresh,
                                                          self._hp.reopen, touch_test or self._prev_touch)
        self._prev_touch = touch_test
        return target