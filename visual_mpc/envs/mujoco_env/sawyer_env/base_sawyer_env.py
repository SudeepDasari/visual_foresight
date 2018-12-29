from visual_mpc.envs.base_env import BaseEnv
from .robosuite_wrappers.SawyerIKEnv import make_sawyer_env
from pyquaternion import Quaternion
import numpy as np
from robosuite.utils.transform_utils import mat2quat, rotation_matrix


class SawyerEnv(BaseEnv):
    def __init__(self, env_params_dict, reset_state=None):
        self._env = make_sawyer_env(env_params_dict)
        self._adim, self._sdim = 5, 5

    def reset(self):
        o = self._env.reset()
        current = self._env._right_hand_orn
        start_rot = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
        pitch, roll, yaw = 0,0,np.pi/2
        drot1 = rotation_matrix(angle=-pitch, direction=[1., 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1., 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.], point=None)[:3, :3]
        desired_rot = start_rot.dot(drot1.dot(drot2.dot(drot3)))
        drotation = current.T.dot(desired_rot)
        import pdb; pdb.set_trace()        
        for _ in range(10):
            dquat = mat2quat(drotation)
            o = self._env.step(np.concatenate(([0,0,0], dquat, [-1])))[0]
            current = self._env._right_hand_orn
            drotation = current.T.dot(desired_rot)
        #import pdb; pdb.set_trace()
        import cv2
        [cv2.imwrite('test_{}.png'.format(i), o['images'][i][:,:,::-1]) for i in range(2)]
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

