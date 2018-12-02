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
        self.low_bound = np.array([-0.5, -0.5, -0.08, -np.pi*2, -1])
        self.high_bound = np.array([0.5, 0.5, 0.15, np.pi*2, 1])
        self._base_adim, self._base_sdim = 5, 6

    def reset(self, reset_state=None):
        obs, write_reset_state = super().reset(reset_state)
        return obs, write_reset_state

    def get_armpos(self, object_pos):
        xpos0 = super().get_armpos(object_pos)
        xpos0[3] = np.random.uniform(-np.pi, np.pi)
        xpos0[4:6] = [0.05, -0.05]

        return xpos0

    def _get_obs(self, finger_sensors):
        obs = super()._get_obs(finger_sensors)
        obs['state'][-1] = self._previous_target_qpos[-1]
        return obs




