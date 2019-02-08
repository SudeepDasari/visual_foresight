from visual_mpc.envs.base_env import BaseEnv
from .robosuite_wrappers.SawyerIKEnv import make_sawyer_env
import numpy as np
from robosuite.utils.transform_utils import mat2quat, rotation_matrix

low_bound = np.array([0.35, -0.2, 0.83, 0, -1])
high_bound = np.array([0.75, 0.2, 0.95, np.pi, 1])
start_rot = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])


class SawyerEnv(BaseEnv):
    def __init__(self, env_params_dict, reset_state=None):
        self._hp = self._default_hparams()
        for name, value in env_params_dict.items():
            print('setting param {} to value {}'.format(name, value))
            self._hp.set_hparam(name, value)
        self._env = make_sawyer_env({'num_objects': self._hp.num_objects})
        self._adim, self._sdim = 5, 5

    def _default_hparams(self):
        parent_params = super()._default_hparams()
        parent_params.add_hparam('substeps', 10)
        parent_params.add_hparam('num_objects', 1)
        return parent_params

    def _init_dynamics(self):
        self._previous_target_qpos = np.random.uniform(low_bound, high_bound)
        self._previous_target_qpos[-1] = low_bound[-1]    # gripper starts open

    def _next_qpos(self, action):
        if action[-1] > 0:
            action[-1] = 1
        else:
            action[-1] = -1
   
        return self._previous_target_qpos * [1., 1., 1., 1., 0.] + action

    def _step(self, target_qpos):
        o = None
        delta_xyz = (target_qpos[:3] - self._eef_pos) / self._hp.substeps
        for i in range(self._hp.substeps):
            current_rot = self._env._right_hand_orn

            pitch, roll, yaw = 0, 0, target_qpos[3]
            drot1 = rotation_matrix(angle=-pitch, direction=[1., 0, 0], point=None)[:3, :3]
            drot2 = rotation_matrix(angle=roll, direction=[0, 1., 0], point=None)[:3, :3]
            drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.], point=None)[:3, :3]
            desired_rot = start_rot.dot(drot1.dot(drot2.dot(drot3)))
            drotation = current_rot.T.dot(desired_rot)
            dquat = mat2quat(drotation)

            o = self._env.step(np.concatenate((delta_xyz, dquat, [target_qpos[-1]])))[0]
        self._previous_target_qpos = target_qpos
        return self._proc_obs(o)

    def _proc_obs(self, env_obs):
        self._eef_pos, self._eef_quat = env_obs['eef_pos'], env_obs['eef_quat']
        return env_obs

    def step(self, action):
        target_qpos = np.clip(self._next_qpos(action), low_bound, high_bound)
        return self._step(target_qpos)

    def reset(self):
        super().reset()
        self._proc_obs(self._env.reset())
        self._init_dynamics()
        return self._step(self._previous_target_qpos), None

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
        return 2

    @property
    def num_objects(self):
        return self._hp.num_objects

