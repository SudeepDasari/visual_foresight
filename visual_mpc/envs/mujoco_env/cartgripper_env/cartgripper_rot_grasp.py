from envs.mujoco_env.cartgripper_env.base_cartgripper import BaseCartgripperEnv
import copy
import numpy as np


class CartgripperRotGraspEnv(BaseCartgripperEnv):
    """
    cartgripper env with motion in x,y,z, rot, grasp
    """
    def __init__(self, env_params, reset_state):
        self._hyper = copy.deepcopy(env_params)
        super().__init__(env_params, reset_state)
        self.low_bound = np.array([-0.5, -0.5, -0.08, -np.pi*2, 0.])
        self.high_bound = np.array([0.5, 0.5, 0.15, np.pi*2, 0.1])
        self._base_adim, self._base_sdim = 5, 6
        self._n_joints = 6
        self._gripper_dim = 4
        self._adim, self._sdim = 5, 5

    def _default_hparams(self):
        parent_params = super()._default_hparams()
        parent_params.set_hparam('filename', 'cartgripper_grasp.xml')
        return parent_params

    def reset(self, reset_state=None):
        obs, write_reset_state = super().reset(reset_state)
        return obs, write_reset_state

    def get_armpos(self, object_pos):
        xpos0 = super().get_armpos(object_pos)
        xpos0[3] = np.random.uniform(-np.pi, np.pi)
        xpos0[4:6] = [0.05, -0.05]

        return xpos0
