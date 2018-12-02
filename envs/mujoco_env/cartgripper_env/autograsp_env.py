from envs.mujoco_env.cartgripper_env.cartgripper_rot_grasp import CartgripperRotGraspEnv
from envs.util.action_util import autograsp_dynamics
from envs.mujoco_env.cartgripper_env.util.sensor_util import is_touching
import copy


class AutograspCartgripperEnv(CartgripperRotGraspEnv):
    def __init__(self, env_params, reset_state=None):
        assert 'mode_rel' not in env_params, "Autograsp sets mode_rel"
        params = copy.deepcopy(env_params)

        if 'autograsp' in params:
            ag_dict = params.pop('autograsp')
            for k in ag_dict:
                params[k] = ag_dict[k]

        super().__init__(params, reset_state)
        self._adim, self._sdim = 4, self._base_sdim

    def _default_hparams(self):
        ag_params = {
            'no_motion_goal': False,
            'reopen': False,
            'zthresh': -0.06,
            'touchthresh': 0.0,
        }

        parent_params = super()._default_hparams()
        parent_params.set_hparam('finger_sensors', True)
        for k in ag_params:
            parent_params.add_hparam(k, ag_params[k])
        return parent_params

    def _init_dynamics(self):
        self._gripper_closed = False

    def _next_qpos(self, action):
        assert action.shape[0] == self._adim
        gripper_z = self._previous_target_qpos[2] + action[2]
        z_thresh = self._hp.zthresh

        target, self._gripper_closed = autograsp_dynamics(self._previous_target_qpos, action,
                                                          self._gripper_closed, gripper_z, z_thresh, self._hp.reopen,
                                                          is_touching(self._last_obs['finger_sensors']))
        return target
