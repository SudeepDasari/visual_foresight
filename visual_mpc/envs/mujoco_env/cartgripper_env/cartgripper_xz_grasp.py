from .base_cartgripper import BaseCartgripperEnv, zangle_to_quat
import numpy as np


class CartgripperXZGrasp(BaseCartgripperEnv):
    def __init__(self, env_params, reset_state = None):
        super().__init__(env_params, reset_state)
        self.low_bound = np.array([-0.4, -0.075, 0])
        self.high_bound = np.array([0.4, 0.15, 0.1])
        self._base_adim, self._base_sdim = 3, 6
        self._adim, self._sdim = 3, 3      # x z grasp
        self._gripper_dim = 2
        self._n_joints = 6

    def _default_hparams(self):
        default_dict = {
            'x_range': 0.3,
            'default_y': 0.,
            'default_theta': 0.,
            'gripper_open': 0.06438482934440347,
            'gripper_close': 0,
            'gripper_thresh': 0.
        }

        parent_params = super()._default_hparams()
        parent_params.set_hparam('filename', 'cartgripper_xz_grasp.xml')
        parent_params.set_hparam('mode_rel', [True, True, False])
        parent_params.set_hparam('finger_sensors', False)
        parent_params.set_hparam('minlen', 0.03)
        parent_params.set_hparam('maxlen', 0.05)
        parent_params.set_hparam('valid_rollout_floor', -2e-1)
        parent_params.set_hparam('ncam', 1)

        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])

        return parent_params

    def _get_state(self):
        gripper_val = (self.sim.data.qpos[4] - self._hp.gripper_close)/(self._hp.gripper_open - self._hp.gripper_close)
        return np.array([self.sim.data.qpos[0], self.sim.data.qpos[2], 1 - gripper_val])

    def _init_dynamics(self):
        self._previous_target_qpos = self._get_state()
        self._goal_reached = False
        self._object_floors = self._last_obs['object_poses_full'].copy()

    def _next_qpos(self, action):
        assert action.shape[0] == self._adim
        action = action.copy()

        grip_action = -1
        if action[-1] > self._hp.gripper_thresh:
            grip_action = 1
        action[-1] = grip_action

        return self._previous_target_qpos * self.mode_rel + action

    def _get_obs(self, finger_sensors):
        base_obs = super()._get_obs(finger_sensors)
        base_obs['state'] = self._get_state()
        self._last_obs['state'] = self._get_state()
        return base_obs

    def _create_pos(self):
        object_poses = super()._create_pos()
        for i in range(self.num_objects):
            object_poses[i][0] = np.random.uniform(-self._hp.x_range, self._hp.x_range)
            object_poses[i][1] = self._hp.default_y
            object_poses[i][3:] = zangle_to_quat(self._hp.default_theta)
        return object_poses

    def get_armpos(self, object_pos):
        xpos0 = np.zeros(self._base_sdim)
        if self.randomize_initial_pos:
            assert not self.arm_obj_initdist
            xpos0[0] = np.random.uniform(-.4, .4)
            xpos0[1] = self._hp.default_y
            xpos0[2] = np.random.uniform(-0.08, .14)
            xpos0[3] = self._hp.default_theta
            xpos0[4:6] = [0.05, -0.05]
        else:
            raise NotImplementedError
        # xpos0[-1] = low_bound[-1]  # start with gripper open
        return xpos0

    def _post_step(self):
        if self._hp.finger_sensors:
            finger_sensors_thresh = np.amax(self._last_obs['finger_sensors']) > 0
        else:
            finger_sensors_thresh = self._last_obs['state'][2] <= 0.9    # check if gripper is closed

        object_deltas = self._last_obs['object_poses_full'][:, 2] - self._object_floors[:, 2]
        z_thresholds = np.amax(object_deltas) >= 0.05 and self._last_obs['state'][1] >= 0.02
        if z_thresholds and finger_sensors_thresh:
            self._goal_reached = True

    def has_goal(self):
        return True

    def goal_reached(self):
        return self._goal_reached

    def _move_arm(self):
        """
        Moves arm to random position
        :return: None
        """
        target_dx = np.random.uniform(-self._hp.x_range, self._hp.x_range) - self._previous_target_qpos[0]
        target_dy = np.random.uniform(0.12, self.high_bound[2]) - self._previous_target_qpos[1]
        self.step(np.array([target_dx, target_dy, -1]))

    def _move_objects(self):
        """
        Creates a lifting task by randomly placing block in gripper until it grasps
            - Randomness needed since there is no "expert" to correctly place object into hand
        :return: None
        """
        i, done = np.random.choice(self.num_objects, 1)[0], False
        block_wiggle = self._hp.maxlen
        while not done:
            target_y = self._previous_target_qpos[1] + 0.015 \
                                                         + np.random.uniform(-block_wiggle, block_wiggle)
            self.sim.data.qpos[self._n_joints + i * 7] = self._previous_target_qpos[0] \
                                                         + np.random.uniform(-block_wiggle, block_wiggle)
            self.sim.data.qpos[self._n_joints + i * 7 + 2] = target_y
            self.sim.step()

            target_cmd = np.array([self._previous_target_qpos[0], self._previous_target_qpos[1], 1])
            for _ in range(self.substeps):
                self.sim.data.qpos[self._n_joints + i * 7 + 2] = target_y
                self.sim.data.ctrl[:] = target_cmd
                self.sim.step()

            for _ in range(self.substeps * 5):
                self.sim.step()

            if self.sim.data.qpos[self._n_joints + i * 7 + 2] > 0.05:
                done = True
            else:
                # open up the fingers and try again
                target_cmd = np.array([self._previous_target_qpos[0], self._previous_target_qpos[1], -1])
                for _ in range(self.substeps):
                    self.sim.data.ctrl[:] = target_cmd
                    self.sim.step()

    def generate_task(self):
        self._move_arm()
        self._move_objects()

    @staticmethod
    def default_ncam():
        return 1
