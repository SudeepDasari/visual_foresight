from visual_mpc.envs.mujoco_env.cartgripper_env.base_cartgripper import BaseCartgripperEnv
import copy


class CartgripperXYZEnv(BaseCartgripperEnv):
    def __init__(self, env_params, reset_state):
        self._hyper = copy.deepcopy(env_params)
        super().__init__(env_params, reset_state)

        self._adim, self._sdim = self._base_adim, self._base_sdim
        self._n_joints = self._adim

    def _default_hparams(self):
        default_dict = {}

        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def _init_dynamics(self):
        return

    def _next_qpos(self, action):
        assert action.shape[0] == self._adim
        return self._previous_target_qpos * self.mode_rel + action