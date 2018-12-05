from .base_sawyer_env import BaseSawyerEnv
import copy


class HumanDemoEnv(BaseSawyerEnv):
    def __init__(self, env_params, _=None):
        self._hyper = copy.deepcopy(env_params)
        BaseSawyerEnv.__init__(self, self._hyper)

    def reset(self):
        BaseSawyerEnv.reset(self)
        while raw_input("Enter 'y' when ready to proceed:") != 'y':
            pass

        return self._get_obs(), None

    def step(self, action):
        raise NotImplementedError("Can't give action to human demo env")