from .base_env import BaseRobotEnv
import copy


class VanillaEnv(BaseRobotEnv):
    def __init__(self, env_params, _=None):
        self._hyper = copy.deepcopy(env_params)
        BaseRobotEnv.__init__(self, env_params)
        self._adim, self._sdim = self._base_adim, self._base_sdim

    def _next_qpos(self, action):
        assert action.shape[0] == self._base_adim, "Action should have shape (5,)"
        return self._previous_target_qpos * self.mode_rel + action
