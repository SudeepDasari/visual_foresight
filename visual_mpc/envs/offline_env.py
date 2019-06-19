from visual_mpc.envs.base_env import BaseEnv


class OfflineSawyerEnv(BaseEnv):
    """
    Emulates a real-image Sawyer Env without access to robot, only works together with the Offline Agent!
    """
    def __init__(self, env_params_dict, reset_state=None):
        self._hp = self._default_hparams()
        self._adim, self._sdim = 5, 5

    def _default_hparams(self):
        default_dict = {}
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def step(self, action):
        """
        return None, the offline agent will append loaded observations
        :param action:
        :return:
        """
        return None

    def reset(self):
        return self.step(None), None

    def valid_rollout(self):
        return True

    @property
    def adim(self):
        return self._adim

    @property
    def sdim(self):
        return self._sdim

    @property
    def ncam(self):
        return 1

    @property
    def num_objects(self):
        return 1

