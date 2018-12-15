from visual_mpc.envs.mujoco_env.cartgripper_env.base_cartgripper import BaseCartgripperEnv
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

    def _move_arm(self):
        """
        Moves arm to random position
        :return: None
        """
        target_dx = np.random.uniform(-0.4, 0.4) - self._previous_target_qpos[0]
        target_dy = np.random.uniform(-0.4, 0.4) - self._previous_target_qpos[1]
        target_dz = np.random.uniform(0.1, self.high_bound[2]) - self._previous_target_qpos[2]
        target_dtheta = np.random.uniform(-np.pi / 2, np.pi / 2) - self._previous_target_qpos[3]

        target_qpos = self._next_qpos(np.array([target_dx, target_dy, target_dz, target_dtheta]))
        target_qpos[-1] = self.low_bound[-1]
        BaseCartgripperEnv._step(self, target_qpos)

    def _move_objects(self):
        """
        Creates a lifting task by randomly placing block in gripper until it grasps
            - Randomness needed since there is no "expert" to correctly place object into hand
        :return: None
        """
        i, done = np.random.choice(self.num_objects, 1)[0], False
        block_wiggle = self._hp.maxlen
        while not done:
            retry = False
            target_z = self._previous_target_qpos[2] + 0.015 \
                                                         + np.random.uniform(-block_wiggle, block_wiggle)
            self.sim.data.qpos[self._n_joints + i * 7] = self._previous_target_qpos[0] \
                                                         + np.random.uniform(-block_wiggle, block_wiggle)
            self.sim.data.qpos[self._n_joints + i * 7 + 1] = self._previous_target_qpos[1] \
                                                         + np.random.uniform(-block_wiggle, block_wiggle)
            self.sim.data.qpos[self._n_joints + i * 7 + 2] = target_z
            self.sim.step()

            target_cmd = np.array([self._previous_target_qpos[0], self._previous_target_qpos[1],
                                   self._previous_target_qpos[2],self._previous_target_qpos[3], self.high_bound[-1]])

            for st in range(self.substeps):
                self.sim.data.qpos[self._n_joints + i * 7 + 2] = target_z
                alpha = min(1.0, 2. * st / self.substeps)
                self.sim.data.ctrl[:] = target_cmd
                self.sim.data.ctrl[-1] = self.low_bound[-1] * (1 - alpha) + self.high_bound[-1] * alpha
                self.sim.step()
                delta_gripper = self.sim.data.qpos[self._n_joints + i * 7:self._n_joints + i * 7 + 2] - \
                                self._previous_target_qpos[:2]
                if np.linalg.norm(delta_gripper) > np.sqrt(2) * self._hp.maxlen:
                    retry = True
                    break


            if not retry:
                for _ in range(self.substeps * 5):
                    self.sim.data.ctrl[:] = target_cmd
                    self.sim.data.ctrl[-1] = self.high_bound[-1]
                    self.sim.step()

                if self.sim.data.qpos[self._n_joints + i * 7 + 2] > 0.01:
                    done = True

            if not done:
                # open up the fingers and try again
                target_cmd = np.array([self._previous_target_qpos[0], self._previous_target_qpos[1],
                                       self._previous_target_qpos[2], self._previous_target_qpos[3],
                                       self.low_bound[-1]])
                for _ in range(self.substeps):
                    self.sim.data.ctrl[:] = target_cmd
                    self.sim.step()

    def generate_task(self):
        self._move_arm()
        self._move_objects()