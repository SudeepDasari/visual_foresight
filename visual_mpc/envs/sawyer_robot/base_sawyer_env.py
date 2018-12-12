from visual_mpc.envs.base_env import BaseEnv
import numpy as np
import random
from geometry_msgs.msg import Quaternion as Quaternion_msg
from pyquaternion import Quaternion
from visual_mpc.agent.general_agent import Image_Exception, Bad_Traj_Exception
from .util.limb_recorder import LimbWSGRecorder
from .util.camera_recorder import CameraRecorder
from .util.impedance_wsg_controller import ImpedanceWSGController, NEUTRAL_JOINT_CMD
from visual_mpc.foresight_rospkg.src.utils import inverse_kinematics
from visual_mpc.envs.util.interpolation import QuinticSpline
import copy
import rospy
import os
from visual_mpc.utils.im_utils import npy_to_mp4
from .util.user_interface import select_points
from .util.topic_utils import IMTopic


CONTROL_RATE = 800
CONTROL_PERIOD = 1. / CONTROL_RATE
INTERP_SKIP = 16


def precalculate_interpolation(p1, p2, duration, last_pos, start_cmd, joint_names):
    spline = QuinticSpline(p1, p2, duration)
    num_queries = int(CONTROL_RATE * duration / INTERP_SKIP) + 1
    jas = []
    last_cmd = start_cmd
    for t in np.linspace(0., duration, num_queries):
        cart_pos = spline.get(t)[0][0]
        interp_pose = state_to_pose(cart_pos[:3], zangle_to_quat(cart_pos[3]))

        try:
            interp_ja = pose_to_ja(interp_pose, last_cmd,
                                   debug_z=cart_pos[3] * 180 / np.pi, retry_on_fail=True)
            last_cmd = interp_ja
            interp_ja = np.array([interp_ja[j] for j in joint_names])
            jas.append(interp_ja)
            last_pos = interp_ja
        except EnvironmentError:
            jas.append(last_pos)
            print('ignoring IK failure')

    interp_ja = []
    for i in range(len(jas) - 1):
        interp_ja.append(jas[i].tolist())
        for j in range(1, INTERP_SKIP):
            t = float(j) / INTERP_SKIP
            interp_point = (1 - t) * jas[i] + t * jas[i + 1]
            interp_ja.append(interp_point.tolist())
    interp_ja.append(jas[-1].tolist())

    return interp_ja


def pix_resize(pix, target_width, original_width):
    return np.round((copy.deepcopy(pix).astype(np.float32) *
              target_width / float(original_width))).astype(np.int64)


def quat_to_zangle(quat):
    """
    :param quat: robot rotation quaternion (assuming rotation around z-axis)
    :return: Rotation angle in z-axis
    """
    angle = (Quaternion(axis=[0, 1, 0], angle=np.pi).inverse * Quaternion(quat)).angle
    if angle < 0:     # pyquaternion calculates in range [-np.pi, np.pi], have to flip to robot range
        return 2 * np.pi + angle
    return angle


def zangle_to_quat(zangle):
    """
    :param zangle in radians
    :return: quaternion
    """
    return (Quaternion(axis=[0, 1, 0], angle=np.pi) * Quaternion(axis=[0, 0, 1], angle= zangle)).elements


def state_to_pose(xyz, quat):
    """
    :param xyz: desired pose xyz
    :param quat: quaternion around z angle in [w, x, y, z] format
    :return: stamped pose
    """
    quat = Quaternion_msg(
        w=quat[0],
        x=quat[1],
        y=quat[2],
        z=quat[3]
    )

    desired_pose = inverse_kinematics.get_pose_stamped(xyz[0],
                                                       xyz[1],
                                                       xyz[2],
                                                       quat)
    return desired_pose


def pose_to_ja(target_pose, start_joints, tolerate_ik_error=False, retry_on_fail = False, debug_z = None):
    try:
        return inverse_kinematics.get_joint_angles(target_pose, seed_cmd=start_joints,
                                                        use_advanced_options=True)
    except ValueError:
        if retry_on_fail:
            print 'retyring zangle was: {}'.format(debug_z)

            return pose_to_ja(target_pose, NEUTRAL_JOINT_CMD)
        elif tolerate_ik_error:
            raise ValueError("IK failure")    # signals to agent it should reset
        else:
            print 'zangle was {}'.format(debug_z)
            raise EnvironmentError("IK Failure")   # agent doesn't handle EnvironmentError


class BaseSawyerEnv(BaseEnv):
    def __init__(self, env_params):
        self._hp = self._default_hparams()
        for name, value in env_params.items():
            print('setting param {} to value {}'.format(name, value))
            if name == 'camera_topics':
                self._hp.camera_topics = value
            else:
                self._hp.set_hparam(name, value)

        print('initializing environment for {}'.format(self._hp.robot_name))
        self._robot_name = self._hp.robot_name
        self._setup_robot()

        if self._hp.opencv_tracking:
            self._obs_tol = 0.5
        else:
            self._obs_tol = self._hp.OFFSET_TOL

        self._controller = ImpedanceWSGController(CONTROL_RATE, self._robot_name,
                                                  self._hp.print_debug, self._hp.gripper_attached)
        self._limb_recorder = LimbWSGRecorder(self._controller)
        self._save_video = self._hp.video_save_dir is not None
        self._cameras = [CameraRecorder(t, self._hp.opencv_tracking, self._save_video) for t in self._hp.camera_topics]

        self._controller.open_gripper(True)
        self._controller.close_gripper(True)
        self._controller.open_gripper(True)

        if len(self._cameras) > 1:
            first_cam_dim = (self._cameras[0].img_height, self._cameras[1].img_width)
            assert all([(c.img_height, c.img_width) == first_cam_dim for c in self._cameras[1:]]), \
                'Camera image streams do not match)'

        if len(self._cameras) == 1:
            self._cam_names = ['front']
        elif len(self._cameras) == 2:
            self._cam_names = ['front', 'left']
        else:
            self._cam_names = ['cam{}'.format(i) for i in range(len(self._cameras))]

        self._height, self._width = self._cameras[0].img_height, self._cameras[0].img_width

        self._base_adim, self._base_sdim = 5, 5
        self._adim, self._sdim, self.mode_rel = None, None, np.array(self._hp.mode_rel)
        self._cleanup_rate, self._duration = self._hp.cleanup_rate, self._hp.duration
        self._reset_counter, self._previous_target_qpos = 0, None

        self._start_pix, self._desig_pix, self._goal_pix = None, None, None
        self._goto_closest_neutral()

    def _default_hparams(self):
        default_dict = {'robot_name': None,
                        'gripper_attached': True,
                        'camera_topics': [IMTopic('/camera0/image_raw', flip=True), IMTopic('/camera1/image_raw')],
                        'opencv_tracking': False,
                        'video_save_dir': None,
                        'start_at_neutral': False,
                        'start_box': [1., 1., 1.],          # amount of xyz state range gripper can start in
                        'OFFSET_TOL': 0.06,
                        'duration': 1.,
                        'mode_rel': [True, True, True, True, False],
                        'lower_bound_delta': [0., 0., 0., 0., 0.],
                        'upper_bound_delta': [0., 0., 0., 0., 0.],
                        'cleanup_rate': 25,
                        'print_debug': True,
                        'rand_drop_reset': True,
                        'normalize_actions': False,
                        'reset_before_eval': False}

        parent_params = BaseEnv._default_hparams(self)
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def _setup_robot(self):
        low_angle = np.pi / 2                  # chosen to maximize wrist rotation given start rotation
        high_angle = 265 * np.pi / 180
        if self._robot_name == 'vestri':                                      # pull the cage a bit backward on vestri
            self._low_bound = np.array([0.47, -0.2, 0.176, low_angle, -1])
            self._high_bound = np.array([0.81, 0.2, 0.292, high_angle, 1])
        elif self._robot_name == 'sudri':
            self._low_bound = np.array([0.45, -0.18, 0.176, low_angle, -1])
            self._high_bound = np.array([0.79, 0.22, 0.292, high_angle, 1])
        elif self._robot_name == 'nordri':
            self._low_bound = np.array([0.45, -0.3, 0.214, low_angle, -1])
            self._high_bound = np.array([0.75, 0.24, 0.33, high_angle, 1])
        else:
            raise ValueError("Supported robots are vestri/sudri")

        self._high_bound += np.array(self._hp.upper_bound_delta, dtype=np.float64)
        self._low_bound += np.array(self._hp.lower_bound_delta, dtype=np.float64)

    def step(self, action):
        """
        Applies the action and steps simulation
        :param action: action at time-step
        :return: obs dict where:
                  -each key is an observation at that step
                  -keys are constant across entire datastep (e.x. every-timestep has 'state' key)
                  -keys corresponding to numpy arrays should have constant shape every timestep (for caching)
                  -images should be placed in the 'images' key in a (ncam, ...) array
        """
        if self._hp.normalize_actions:
            action[:3] *= self._high_bound[:3] - self._low_bound[:3]

        target_qpos = np.clip(self._next_qpos(action), self._low_bound, self._high_bound)

        if np.linalg.norm(target_qpos - self._previous_target_qpos) < 1e-3:
            return self._get_obs()

        wait_change = (target_qpos[-1] > 0) != (self._previous_target_qpos[-1] > 0)

        if self._save_video:
            [c.start_recording() for c in self._cameras]

        if target_qpos[-1] > 0:
            self._controller.close_gripper(wait_change)
        else:
            self._controller.open_gripper(wait_change)

        self._move_to_state(target_qpos[:3], target_qpos[3])

        if self._save_video:
            [c.stop_recording() for c in self._cameras]

        self._previous_target_qpos = target_qpos
        return self._get_obs()

    def _init_dynamics(self):
        """
        Initializes custom dynamics for action space
        :return: None
        """
        pass

    def _next_qpos(self, action):
        """
        Next target state given current state/actions
        :return: next_state
        """
        raise NotImplementedError

    def _get_state(self):
        j_angles, j_vel, eep, gripper_state, force_sensor = self._limb_recorder.get_state()
        state = np.zeros(self._base_sdim)
        state[:3] = (eep[:3] - self._low_bound[:3]) / (self._high_bound[:3] - self._low_bound[:3])
        state[3] = quat_to_zangle(eep[3:])
        state[4] = gripper_state * self._low_bound[-1] + (1 - gripper_state) * self._high_bound[-1]
        return state

    def _get_obs(self):
        obs = {}
        j_angles, j_vel, eep, gripper_state, force_sensor = self._limb_recorder.get_state()
        obs['qpos'] = j_angles
        obs['qvel'] = j_vel

        if self._hp.print_debug and self._previous_target_qpos is not None:
            print 'xy delta: ', np.linalg.norm(eep[:2] - self._previous_target_qpos[:2])
            print 'target z', self._previous_target_qpos[2], 'real z', eep[2]
            print 'z dif', abs(eep[2] - self._previous_target_qpos[2])
            print 'angle dif (degrees): ', abs(quat_to_zangle(eep[3:]) - self._previous_target_qpos[3]) * 180 / np.pi
            print 'angle degree target {} vs real {}'.format(np.rad2deg(quat_to_zangle(eep[3:])),
                                                             np.rad2deg(self._previous_target_qpos[3]))

        state = np.zeros(self._base_sdim)
        state[:3] = (eep[:3] - self._low_bound[:3]) / (self._high_bound[:3] - self._low_bound[:3])
        state[3] = quat_to_zangle(eep[3:])
        state[4] = gripper_state * self._low_bound[-1] + (1 - gripper_state) * self._high_bound[-1]
        obs['state'] = state
        obs['finger_sensors'] = force_sensor

        self._last_obs = copy.deepcopy(obs)
        obs['images'] = self.render()
        obs['high_bound'],obs['low_bound'] = copy.deepcopy(self._high_bound), copy.deepcopy(self._low_bound)

        if self._hp.opencv_tracking:
            track_desig = np.zeros((self.ncam, 1, 2), dtype=np.int64)
            for i, c in enumerate(self._cameras):
                track_desig[i] = c.get_track()
            self._desig_pix = track_desig

        if self._desig_pix is not None:
            obs['obj_image_locations'] = copy.deepcopy(self._desig_pix)
        return obs

    def _get_xyz_angle(self):
        cur_xyz, cur_quat = self._limb_recorder.get_xyz_quat()
        return cur_xyz, quat_to_zangle(cur_quat)

    def _move_to_state(self, target_xyz, target_zangle, duration = None):
        if duration is None:
            duration = self._duration
        p1 = np.zeros(4)
        p1[:3], p1[3] = self._get_xyz_angle()
        p2 = np.zeros(4)
        p2[:3], p2[3] = target_xyz, target_zangle

        last_pos = self._limb_recorder.get_joint_angles()
        last_cmd = self._limb_recorder.get_joint_cmd()
        joint_names = self._limb_recorder.get_joint_names()

        interp_jas = precalculate_interpolation(p1, p2, duration, last_pos, last_cmd, joint_names)

        i = 0
        self._controller.control_rate.sleep()
        start_time = rospy.get_time()
        t = rospy.get_time()
        while t - start_time < duration:
            lookup_index = min(int(min((t - start_time), duration) / CONTROL_PERIOD), len(interp_jas) - 1)
            self._controller.send_pos_command(interp_jas[lookup_index])
            i += 1
            self._controller.control_rate.sleep()
            t = rospy.get_time()
        if self._hp.print_debug:
            print('Effective rate: {} Hz'.format(i / (rospy.get_time() - start_time)))

    def _reset_previous_qpos(self):
        eep = self._limb_recorder.get_state()[2]
        self._previous_target_qpos = np.zeros(self._base_sdim)
        self._previous_target_qpos[:3] = eep[:3]
        self._previous_target_qpos[3] = quat_to_zangle(eep[3:])
        self._previous_target_qpos[4] = -1

    def _save_videos(self):
        if self._save_video:
            buffers = [c.reset_recording() for c in self._cameras]
            if max([len(b) for b in buffers]) == 0:
                return

            clip_base_name = '{}/recording{}/'.format(self._hp.video_save_dir, self._reset_counter)
            if not os.path.exists:
                os.makedirs(clip_base_name)

            for name, b in zip(self._cam_names, buffers):
                if len(b) > 0:
                    npy_to_mp4(b, '{}/{}_clip'.format(clip_base_name, name), 30)

    def _end_reset(self):
        if self._hp.opencv_tracking:
            assert self._desig_pix is not None, "Designated pixels must be set (call get_obj_desig_goal)"
            track_desig = copy.deepcopy(self._desig_pix)
            [c.start_tracking(track_desig[i]) for i, c in enumerate(self._cameras)]

        self._reset_previous_qpos()
        self._init_dynamics()
        self._reset_counter += 1
        return self._get_obs(), None

    def _goto_closest_neutral(self):
        self._controller.neutral_with_impedance()
        closest_netural = self._get_state()

        closest_netural[:3] = np.clip(closest_netural[:3], [0., 0., 0.], self._hp.start_box)
        closest_netural[:3] *= self._high_bound[:3] - self._low_bound[:3]
        closest_netural[:3] += self._low_bound[:3]

        self._move_to_state(closest_netural[:3], closest_netural[3], 1.)

    def reset(self):
        """
        Resets the environment and returns initial observation
        :return: obs dict (look at step(self, action) for documentation)
        """
        self._save_videos()

        if self._hp.start_at_neutral:
            self._controller.open_gripper(True)
            self._goto_closest_neutral()
            return self._end_reset()

        if self._hp.rand_drop_reset:
            rand_xyz = np.random.uniform(self._low_bound[:3], self._high_bound[:3])
            rand_xyz[2] = self._high_bound[2]
            rand_zangle = np.random.uniform(self._low_bound[3], self._high_bound[3])
            self._move_to_state(rand_xyz, rand_zangle, 2.)
            self._controller.close_gripper(True)
            self._controller.open_gripper(True)
            self._controller.neutral_with_impedance()
        else:
            self._controller.open_gripper(True)
            self._controller.neutral_with_impedance()

        if self._cleanup_rate > 0 and self._reset_counter % self._cleanup_rate == 0 and self._reset_counter > 0:
            self._controller.redistribute_objects()

        self._controller.neutral_with_impedance()
        self._controller.open_gripper(False)
        rospy.sleep(0.5)
        self._reset_previous_qpos()


        rand_xyz = np.random.uniform(self._low_bound[:3], self._high_bound[:3])
        rand_zangle = np.random.uniform(self._low_bound[3], self._high_bound[3])
        self._move_to_state(rand_xyz, rand_zangle, 2.)

        return self._end_reset()

    def valid_rollout(self):
        """
        Checks if the environment is currently in a valid state
        Common invalid states include:
            - object falling out of bin
            - mujoco error during rollout
        :return: bool value that is False if rollout isn't valid
        """
        return True

    def goal_reached(self):
        """
        Checks if the environment hit a goal (if environment has goals)
            - e.x. if goal is to lift object should return true if object lifted by gripper
        :return: whether or not environment reached goal state
        """
        raise NotImplementedError("Environment has No Goal")

    def has_goal(self):
        """
        :return: Whether or not environment has a goal
        """
        return False

    def render(self):
        """ Grabs images form cameras.
        If returning multiple images asserts timestamps are w/in OBS_TOLERANCE, and raises Image_Exception otherwise

        - dual: renders both left and main cameras
        - left: renders only left camera
        - main: renders only main (front) camera
        :param mode: Mode to render with (dual by default)
        :return: uint8 numpy array with rendering from sim
        """
        time_stamps = []
        cam_imgs = []
        cur_time = rospy.get_time()

        for recorder in self._cameras:
            stamp, image = recorder.get_image()
            if abs(stamp - cur_time) > 10 * self._obs_tol:    # no camera ping in half second => camera failure
                print("DeSYNC!")
                raise Image_Exception
            time_stamps.append(stamp)
            cam_imgs.append(image)

        for index, i in enumerate(time_stamps[:-1]):
            for j in time_stamps[index + 1:]:
                if abs(i - j) > self._obs_tol:
                    print('DeSYNC!')
                    raise Image_Exception

        images = np.zeros((self.ncam, self._height, self._width, 3), dtype=np.uint8)
        for c, img in enumerate(cam_imgs):
            images[c] = img[:, :, ::-1]

        return images

    @property
    def adim(self):
        """
        :return: Environment's action dimension
        """
        return self._adim

    @property
    def sdim(self):
        """
        :return: Environment's state dimension
        """
        return self._sdim

    @property
    def ncam(self):
        """
        Sawyer environment has ncam cameras
        """
        return len(self._cameras)

    @property
    def num_objects(self):
        """
        :return: Dummy value for num_objects (used in general_agent logic)
        """
        return 0

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def eval(self, target_width=None, save_dir=None, ntasks=None):
        if target_width == None:
            return None

        self._save_videos()
        if self._hp.reset_before_eval:
            self._controller.open_gripper(True)
            self._controller.neutral_with_impedance()

        final_pix = select_points(self.render(), ['front', 'left'], 'final',
                                 save_dir, clicks_per_desig=1, n_desig=ntasks)

        goal_pix = self.get_goal_pix(target_width)
        final_pix = pix_resize(final_pix, target_width, self._width)
        start_pix = pix_resize(self._start_pix, target_width, self._width)

        final_dist, start_dist = np.linalg.norm(final_pix - goal_pix), np.linalg.norm(start_pix - goal_pix)
        improvement = start_dist - final_dist
        print 'final_dist: {}'.format(final_dist)
        print 'start dist: {}'.format(start_dist)
        print 'improvement: {}'.format(improvement)

        if self._hp.opencv_tracking:
            [c.end_tracking() for c in self._cameras]

        return {'final_dist': final_dist, 'start_dist': start_dist, 'improvement': improvement}

    def get_obj_desig_goal(self, save_dir, collect_goal_image=False, ntasks=1):
        if self._hp.video_save_dir is not None:
            self._hp.video_save_dir = save_dir

        raw_input("Robot in safe position? Hit enter when ready...")
        self._goto_closest_neutral()
        self._controller.open_gripper(True)

        if collect_goal_image:
            print("PLACE OBJECTS IN GOAL POSITION")
            raw_input("When ready to annotate GOAL images press enter...")
            goal_imgs = self.render()
            goal_pix = select_points(goal_imgs, ['front', 'left'], 'goal',
                                     save_dir, clicks_per_desig=1, n_desig=ntasks)

            raw_input("Robot in safe position? Hit enter when ready...")
            self._goto_closest_neutral()
            self._controller.open_gripper(True)

            print("PLACE OBJECTS IN START POSITION")
            raw_input("When ready to annotate START images press enter...")

            self._start_pix = select_points(self.render(), ['front', 'left'], 'desig',
                                     save_dir, clicks_per_desig=1, n_desig=ntasks)
            self._goal_pix = copy.deepcopy(goal_pix)
            self._desig_pix = copy.deepcopy(self._start_pix)

            return goal_imgs, goal_pix
        else:
            print("PLACE OBJECTS IN START POSITION")
            raw_input("When ready to annotate START images press enter...")

            self._start_pix, self._goal_pix = select_points(self.render(), ['front', 'left'], 'desig_goal',
                                     save_dir, n_desig=ntasks)
            self._desig_pix = copy.deepcopy(self._start_pix)
            return copy.deepcopy(self._goal_pix)

    def get_goal_pix(self, target_width):
        return pix_resize(self._goal_pix, target_width, self._width)
