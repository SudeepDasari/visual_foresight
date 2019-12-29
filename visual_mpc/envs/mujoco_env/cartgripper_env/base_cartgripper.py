from visual_mpc.envs.mujoco_env.base_mujoco_env import BaseMujocoEnv
import numpy as np
import visual_mpc.envs as envs
from visual_mpc.envs.mujoco_env.util.create_xml import create_object_xml, create_root_xml, clean_xml
import copy
from pyquaternion import Quaternion
from visual_mpc.utils.im_utils import npy_to_mp4


BASE_DIR = '/'.join(str.split(envs.__file__, '/')[:-1])
asset_base_path = BASE_DIR + '/mjc_models/cartgripper_assets/'
low_bound = np.array([-0.5, -0.5, -0.08, -np.pi*2, 0.])
high_bound = np.array([0.5, 0.5, 0.15, np.pi*2, 0.1])
is_open_thresh = 0.5 * (low_bound[-1] + high_bound[-1])


def zangle_to_quat(zangle):
    """
    :param zangle in rad
    :return: quaternion
    """
    return np.array([np.cos(zangle/2), 0, 0, np.sin(zangle/2)])


def quat_to_zangle(quat):
    """
    :param quat: quaternion with only
    :return: zangle in rad
    """
    theta = np.arctan2(2 * quat[0] * quat[3], 1 - 2 * quat[3] ** 2)
    return np.array([theta])


class BaseCartgripperEnv(BaseMujocoEnv):

    """
    cartgripper env with motion in x,y,z
    """
    def __init__(self, env_params_dict, reset_state = None):
        params_dict = copy.deepcopy(env_params_dict)
        #TF HParams can't handle list Hparams well, this is cleanest workaround for object_meshes
        if 'object_meshes' in params_dict:
            object_meshes = params_dict.pop('object_meshes')
        else:
            object_meshes = None

        _hp = self._default_hparams()
        for name, value in params_dict.items():
            print('setting param {} to value {}'.format(name, value))
            _hp.set_hparam(name, value)

        base_filename = asset_base_path + _hp.filename
        friction_params = (_hp.friction, 0.010, 0.0002)
        reset_xml = None
        if reset_state is not None:
            reset_xml = reset_state['reset_xml']
        self._reset_xml = create_object_xml(base_filename, _hp.num_objects, _hp.object_mass,
                                            friction_params, object_meshes, _hp.finger_sensors,
                                            _hp.maxlen, _hp.minlen, reset_xml,
                                            _hp.obj_classname, cube_objs=_hp.cube_objects,
                                            block_height=_hp.block_height)
        gen_xml = create_root_xml(base_filename)
        super().__init__(gen_xml, _hp)
        if _hp.clean_xml:
            clean_xml(gen_xml)

        self._base_sdim, self._base_adim, self.mode_rel = 3, 3, np.array(_hp.mode_rel)
        self.num_objects, self.skip_first, self.substeps = _hp.num_objects, _hp.skip_first, _hp.substeps
        self.sample_objectpos = _hp.sample_objectpos
        self.object_object_mindist = _hp.object_object_mindist
        self.randomize_initial_pos = _hp.randomize_initial_pos
        self.arm_obj_initdist = _hp.arm_obj_initdist
        self.arm_start_lifted = _hp.arm_start_lifted
        self.finger_sensors, self.object_sensors = _hp.finger_sensors, object_meshes is not None
        self._previous_target_qpos, self._n_joints = None, 3
        self._hp = _hp

        self._read_reset_state = reset_state
        self.low_bound = np.array([-0.5, -0.5, -0.08])
        self.high_bound = np.array([0.5, 0.5, 0.15])
        self._gripper_dim = None

    def _default_hparams(self):
        default_dict = {
                          'verbose':False,
                          'filename': 'cartgripper_updown_2cam.xml',
                          'num_objects': 1,
                          'object_mass': 0.1,
                          'friction':1.,
                          'mode_rel': [True, True, True],
                          'object_meshes':None,
                          'finger_sensors':False,
                          'maxlen': 0.2,
                          'minlen': 0.01,
                          'preload_obj_dict': None,
                          'sample_objectpos':True,
                          'object_object_mindist':0.,
                          'randomize_initial_pos': True,
                          'arm_obj_initdist': None,
                          'xpos0': None,
                          'object_pos0': np.array([]),
                          'arm_start_lifted': True,
                          'skip_first': 40,
                          'obj_classname':None,
                          'substeps': 500,
                          'clean_xml': True,
                          'cube_objects': False,
                          'block_height': 0.03,
                          'valid_rollout_floor': -2e-2,
                          'use_vel':False,}

        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def _step(self, target_qpos):
        assert target_qpos.shape[0] == self._base_adim
        finger_force = np.zeros(2)

        for st in range(self.substeps):
            if self.finger_sensors:
                finger_force += copy.deepcopy(self.sim.data.sensordata[:2].squeeze())

            alpha = st / (float(self.substeps) - 1)
            self.sim.data.ctrl[:] = alpha * target_qpos + (1. - alpha) * self._previous_target_qpos
            self.sim.step()

        finger_force /= self.substeps

        self._previous_target_qpos = target_qpos
        obs = self._get_obs(finger_force)
        self._post_step()

        return obs

    def step(self, action):
        target_qpos = np.clip(self._next_qpos(action), self.low_bound, self.high_bound)
        return self._step(target_qpos)

    def _post_step(self):
        return

    def render(self):
        return super().render()[:, ::-1].copy()    # cartgripper cameras are flipped in height dimension

    def project_point(self, point, camera):
        row, col = super().project_point(point, camera)
        return self._frame_height - row, col      # cartgripper cameras are flipped in height dimension

    def qpos_reset(self, qpos, qvel):
        self._read_reset_state['qpos_all'] = qpos
        self._read_reset_state['qvel_all'] = qvel
        return self.reset(self._read_reset_state)

    def _create_pos(self):
        if self.object_object_mindist > 0:
            min_dist = self.object_object_mindist
        else:
            min_dist = 0.                         # practically inf distance

        # randomly sample initial configurations, if min_dist set find one where objects are at least min_dist apart
        attempts, poses, max_attempts = 0, [], 3000000
        while attempts < max_attempts:
            poses = []

            for i in range(self.num_objects):
                pos = np.random.uniform(-.35, .35, 2)

                if attempts < (max_attempts - 1) and i > 0:
                    if min([np.linalg.norm(pos - p[:2]) for p in poses]) < min_dist:
                        break

                ori = zangle_to_quat(np.random.uniform(0, np.pi * 2))
                poses.append(np.concatenate((pos, np.array([0]), ori), axis=0))

            if len(poses) == self.num_objects:
                break
            attempts += 1

        if attempts == max_attempts - 1:
            print("WARNING COULDN'T SPACE OBJECTS: MIN_DIST MAY BE SET TOO HIGH")
        return poses

    def reset(self, reset_state=None):
        super().reset()

        if reset_state is not None:
            self._read_reset_state = reset_state

        write_reset_state = {}
        write_reset_state['reset_xml'] = copy.deepcopy(self._reset_xml)

        #clear our observations from last rollout
        self._last_obs = None

        if self._read_reset_state is None:
            # create random starting poses for objects
            object_pos_l = self._create_pos()
            object_pos = np.concatenate(object_pos_l)

            # determine arm position
            xpos0 = self.get_armpos(object_pos)
            qpos = np.concatenate((xpos0, object_pos.flatten()), 0)
        else:
            qpos = self._read_reset_state['qpos_all']

        sim_state = self.sim.get_state()
        sim_state.qpos[:] = qpos
        sim_state.qvel[:] = np.zeros_like(self.sim.data.qvel)

        self.sim.set_state(sim_state)
        self.sim.forward()
        write_reset_state['qpos_all'] = qpos
        finger_force = np.zeros(2)
        for t in range(self.skip_first):
            for _ in range(self.substeps):
                self.sim.data.ctrl[:] = qpos[:self._base_adim]
                if self._gripper_dim:
                    self.sim.data.ctrl[self._gripper_dim] = 0.
                self.sim.step()
                if self.finger_sensors:
                    finger_force += copy.deepcopy(self.sim.data.sensordata[:2].squeeze())

        self._previous_target_qpos = copy.deepcopy(self.sim.data.qpos[:self._base_adim].squeeze())
        self._previous_target_qpos[-1] = self.low_bound[-1]
        reset_obs = self._get_obs(finger_force / self.skip_first / self.substeps)

        self._init_dynamics()
        self._reset_eval()

        return reset_obs, write_reset_state

    def get_armpos(self, object_pos):
        xpos0 = np.zeros(self._base_sdim)
        if self.randomize_initial_pos:
            assert not self.arm_obj_initdist
            xpos0[:2] = np.random.uniform(-.4, .4, 2)
            xpos0[2] = np.random.uniform(-0.08, .14)
        elif self.arm_obj_initdist:
            d = self.arm_obj_initdist
            alpha = np.random.uniform(-np.pi, np.pi, 1)
            delta_pos = np.array([d * np.cos(alpha), d * np.sin(alpha)])
            xpos0[:2] = object_pos[:2] + delta_pos.squeeze()
            xpos0[2] = np.random.uniform(-0.08, .14)
        else:
            xpos0 = self._read_reset_state['state']
        if self.arm_start_lifted:
            xpos0[2] = 0.14
        # xpos0[-1] = low_bound[-1]  # start with gripper open
        return xpos0

    def _append_save_buffer(self, img):
        super()._append_save_buffer(img[::-1])

    def _get_obs(self, finger_sensors):
        obs, touch_offset = {}, 0
        #report finger sensors as needed
        if self.finger_sensors:
            obs['finger_sensors'] = finger_sensors
            touch_offset = 2

        #joint poisitions and velocities
        obs['qpos'] = copy.deepcopy(self.sim.data.qpos[:self._n_joints].squeeze())
        obs['qpos_full'] = copy.deepcopy(self.sim.data.qpos)
        obs['qvel'] = copy.deepcopy(self.sim.data.qvel[:self._n_joints].squeeze())
        obs['qvel_full'] = copy.deepcopy(self.sim.data.qvel.squeeze())

        if self._hp.use_vel:
            obs['state'] = np.concatenate([copy.deepcopy(self.sim.data.qpos[:self._sdim].squeeze()),
                                           copy.deepcopy(self.sim.data.qvel[:self._sdim].squeeze())])
        else:
            obs['state'] = copy.deepcopy(self.sim.data.qpos[:self._sdim].squeeze())

        if self._gripper_dim and self._previous_target_qpos[-1] < is_open_thresh:
            obs['state'][self._gripper_dim] = -1
        else:
            obs['state'][self._gripper_dim] = 1

        #report object poses
        obs['object_poses_full'] = np.zeros((self.num_objects, 7))
        obs['object_qpos'] = np.zeros((self.num_objects, 7))
        obs['object_poses'] = np.zeros((self.num_objects, 3))

        for i in range(self.num_objects):
            pos_sen = copy.deepcopy(self.sim.data.sensordata[touch_offset + i * 3:touch_offset + (i + 1) * 3])
            fullpose = copy.deepcopy(self.sim.data.qpos[i * 7 + self._n_joints:(i + 1) * 7 + self._n_joints].squeeze())
            fullpose[:3] = pos_sen
            obs['object_poses_full'][i] = fullpose

            obs['object_poses'][i, :2] = pos_sen[:2]
            obs['object_poses'][i, 2] = Quaternion(fullpose[3:]).angle
            obs['object_qpos'][i] = copy.deepcopy(self.sim.data.qpos[self._n_joints + i * 7: self._n_joints + (i+1)*7])

        #copy non-image data for environment's use (if needed)
        self._last_obs = copy.deepcopy(obs)

        #get images
        obs['images'] = self.render()
        obs['obj_image_locations'] = self.get_desig_pix(self._frame_width, obj_poses=obs['object_poses_full'])

        return obs

    def valid_rollout(self):
        object_zs = self._last_obs['object_poses_full'][:, 2]
        return not any(object_zs < self._hp.valid_rollout_floor)

    def _init_dynamics(self):
        raise NotImplementedError

    def _next_qpos(self, action):
        raise NotImplementedError

    def move_arm(self):
        pass

    def move_objects(self):
        """
        move objects randomly, used to create startgoal-configurations
        """

        def get_new_obj_pose(curr_pos, curr_quat):
            angular_disp = 0.0
            delta_alpha = np.random.uniform(-angular_disp, angular_disp)
            delta_rot = Quaternion(axis=(0.0, 0.0, 1.0), radians=delta_alpha)
            curr_quat = Quaternion(curr_quat)
            newquat = delta_rot * curr_quat

            pos_ok = False
            while not pos_ok:
                const_dist = True
                if const_dist:
                    alpha = np.random.uniform(-np.pi, np.pi, 1)
                    d = 0.25
                    delta_pos = np.array([d * np.cos(alpha), d * np.sin(alpha), 0.])
                else:
                    pos_disp = 0.1
                    delta_pos = np.concatenate([np.random.uniform(-pos_disp, pos_disp, 2), np.zeros([1])])
                newpos = curr_pos + delta_pos
                lift_object = False
                if lift_object:
                    newpos[2] = 0.15
                if np.any(newpos[:2] > high_bound[:2]) or np.any(newpos[:2] < low_bound[:2]):
                    pos_ok = False
                else:
                    pos_ok = True

            return newpos, newquat

        for i in range(self.num_objects):
            curr_pos = self.sim.data.qpos[self._n_joints + i * 7: self._n_joints + 3 + i * 7]
            curr_quat = self.sim.data.qpos[self._n_joints + 3 + i * 7: self._n_joints + 7 + i * 7]
            obji_xyz, obji_quat = get_new_obj_pose(curr_pos, curr_quat)
            self.sim.data.qpos[self._n_joints + i * 7: self._n_joints + 3 + i * 7] = obji_xyz
            self.sim.data.qpos[self._n_joints + 3 + i * 7: self._n_joints + 7 + i * 7] = obji_quat.elements

        sim_state = self.sim.get_state()
        # sim_state.qpos[:] = sim_state.qpos
        sim_state.qvel[:] = np.zeros_like(sim_state.qvel)
        self.sim.set_state(sim_state)
        self.sim.forward()

    def snapshot_noarm(self):
        qpos = copy.deepcopy(self.sim.data.qpos)
        qpos[2] -= 10
        sim_state = self.sim.get_state()
        sim_state.qpos[:] = qpos
        self.sim.set_state(sim_state)
        self.sim.forward()
        image = self.render('maincam').squeeze()
        qpos[2] += 10
        sim_state.qpos[:] = qpos
        self.sim.set_state(sim_state)
        self.sim.forward()

        return image

    def current_obs(self):
        finger_force = np.zeros(2)
        if self.finger_sensors:
            finger_force += self.sim.data.sensordata[:2]
        return self._get_obs(finger_force)
