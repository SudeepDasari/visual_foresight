from visual_mpc.envs.base_env import BaseEnv
from .robosuite_wrappers.SawyerIKEnv import make_sawyer_env


class SawyerEnv(BaseEnv):
    def __init__(self, env_params_dict, reset_state=None):
        self._env = make_sawyer_env(env_params_dict)
        self._adim, self._sdim = 5, 5

    def reset(self):
        o = self._env.reset()
        import pdb
        pdb.set_trace()

    def step(self, action):
        return None

    @property
    def adim(self):
        return self._adim

    @property
    def sdim(self):
        return self._sdim

    @property
    def ncam(self):
        return 2

    @property
    def num_objects(self):
        return 1

