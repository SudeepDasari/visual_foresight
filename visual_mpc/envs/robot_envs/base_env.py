from visual_mpc.envs.robot_envs import get_controller_class
from visual_mpc.envs.base_env import BaseEnv
import numpy as np
import random
from visual_mpc.agent.general_agent import Image_Exception
from .util.camera_recorder import CameraRecorder
import copy
import rospy
from .util.user_interface import select_points
from .util.topic_utils import IMTopic
import logging


def pix_resize(pix, target_width, original_width):
    return np.round((copy.deepcopy(pix).astype(np.float32) *
              target_width / float(original_width))).astype(np.int64)


class BaseRobotEnv(BaseEnv):
    def __init__(self, env_params):
        self._hp = self._default_hparams()
        self._hp.start_state = []
        for name, value in env_params.items():
            if name == 'camera_topics':
                self._hp.camera_topics = value
            elif name == 'start_state':
                self._hp.start_state = value
            else:
                self._hp.set_hparam(name, value)

        logging.info('initializing environment for {}'.format(self._hp.robot_name))
        self._robot_name = self._hp.robot_name
        self._setup_robot()

        if self._hp.opencv_tracking:
            self._obs_tol = 0.5
        else:
            self._obs_tol = self._hp.OFFSET_TOL

        RobotController = get_controller_class(self._hp.robot_type)
        self._controller = RobotController(self._robot_name, self._hp.print_debug, email_cred_file=self._hp.email_login_creds, 
                                           log_file=self._hp.log_file, gripper_attached=self._hp.gripper_attached)
        logging.getLogger('robot_logger').info('---------------------------------------------------------------------------')
        for name, value in self._hp.values().items():
            logging.getLogger('robot_logger').info('{}= {}'.format(name, value))
        logging.getLogger('robot_logger').info('---------------------------------------------------------------------------')

        self._save_video = self._hp.save_video
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
                        'robot_type': 'sawyer',
                        'email_login_creds': '',
                        'log_file': '',
                        'gripper_attached': 'wsg-50',
                        'camera_topics': [IMTopic('/camera0/image_raw', flip=True), IMTopic('/camera1/image_raw')],
                        'opencv_tracking': False,
                        'save_video': False,
                        'start_at_neutral': False,
                        'start_box': [1., 1., 1.],          # amount of xyz state range gripper can start in
                        'OFFSET_TOL': 0.06,
                        'duration': 1.,
                        'mode_rel': [True, True, True, True, False],
                        'lower_bound_delta': [0., 0., 0., 0., 0.],
                        'upper_bound_delta': [0., 0., 0., 0., 0.],
                        'cleanup_rate': 25,
                        'print_debug': False,
                        'rand_drop_reset': True,
                        'normalize_actions': False,
                        'reset_before_eval': False,
                        'wait_during_resetend': False}

        parent_params = BaseEnv._default_hparams(self)
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def _setup_robot(self):
        low_angle = np.pi / 2                  # chosen to maximize wrist rotation given start rotation
        high_angle = 265 * np.pi / 180
        
        # make a more extensible way to do this
        if self._robot_name == 'vestri':                                      # pull the cage a bit backward on vestri
            self._low_bound = np.array([0.47, -0.2, 0.176, low_angle, -1])
            self._high_bound = np.array([0.81, 0.2, 0.292, high_angle, 1])
        elif self._robot_name == 'vestri_table':
            self._low_bound = np.array([0.43, -0.34, 0.17, low_angle, -1])
            self._high_bound = np.array([0.89, 0.32, 0.286, high_angle, 1])
        elif self._robot_name == 'sudri':
            self._low_bound = np.array([0.45, -0.18, 0.176, low_angle, -1])
            self._high_bound = np.array([0.79, 0.22, 0.292, high_angle, 1])
        elif self._robot_name == 'nordri':
            self._low_bound = np.array([0.45, -0.3, 0.214, low_angle, -1])
            self._high_bound = np.array([0.75, 0.24, 0.33, high_angle, 1])
        elif self._robot_name == 'test':
            self._low_bound = np.array([0.47, -0.2, 0.1587, low_angle, -1])
            self._high_bound = np.array([0.81, 0.2, 0.2747, high_angle, 1])
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
        eep = self._controller.get_cartesian_pose()
        gripper_state = self._controller.get_gripper_state()[0]
        
        state = np.zeros(self._base_sdim)
        state[:3] = (eep[:3] - self._low_bound[:3]) / (self._high_bound[:3] - self._low_bound[:3])
        state[3] = self._controller.quat_2_euler(eep[3:])[0]
        state[4] = gripper_state * self._low_bound[-1] + (1 - gripper_state) * self._high_bound[-1]
        return state

    def _get_obs(self):
        obs = {}
        j_angles, j_vel, eep = self._controller.get_state()
        gripper_state, force_sensor = self._controller.get_gripper_state()

        z_angle = self._controller.quat_2_euler(eep[3:])[0]

        obs['qpos'] = j_angles
        obs['qvel'] = j_vel

        if self._previous_target_qpos is not None:
            logging.getLogger('robot_logger').debug('xy delta: {}'.format(np.linalg.norm(eep[:2] - self._previous_target_qpos[:2])))
            logging.getLogger('robot_logger').debug('target z: {}       real z: {}'.format(self._previous_target_qpos[2], eep[2]))   
            logging.getLogger('robot_logger').debug('z dif {}'.format(abs(eep[2] - self._previous_target_qpos[2])))
            logging.getLogger('robot_logger').debug('angle dif (degrees): {}'.format(abs(z_angle - self._previous_target_qpos[3]) * 180 / np.pi))
            logging.getLogger('robot_logger').debug('angle degree target {} vs real {}'.format(np.rad2deg(z_angle),
                                                             np.rad2deg(self._previous_target_qpos[3])))

        obs['state'] = self._get_state()
        obs['finger_sensors'] = force_sensor

        self._last_obs = copy.deepcopy(obs)
        obs['images'] = self.render()
        obs['high_bound'], obs['low_bound'] = copy.deepcopy(self._high_bound), copy.deepcopy(self._low_bound)

        if self._hp.opencv_tracking:
            track_desig = np.zeros((self.ncam, 1, 2), dtype=np.int64)
            for i, c in enumerate(self._cameras):
                track_desig[i] = c.get_track()
            self._desig_pix = track_desig

        if self._desig_pix is not None:
            obs['obj_image_locations'] = copy.deepcopy(self._desig_pix)
        return obs

    def _move_to_state(self, target_xyz, target_zangle, duration = None):
        target_quat = self._controller.euler_2_quat(target_zangle)
        self._controller.move_to_eep(np.concatenate((target_xyz, target_quat)))

    def _reset_previous_qpos(self):
        xyz, quat = self._controller.get_xyz_quat()

        self._previous_target_qpos = np.zeros(self._base_sdim)
        self._previous_target_qpos[:3] = xyz
        self._previous_target_qpos[3] = self._controller.quat_2_euler(quat)[0]
        self._previous_target_qpos[4] = -1

    def save_recording(self, save_worker, i_traj):
        if self._save_video:
            buffers = [c.reset_recording() for c in self._cameras]
            if max([len(b) for b in buffers]) == 0:
                return

            for name, b in zip(self._cam_names, buffers):
                save_worker.put(('mov', 'recording{}/{}_clip.mp4'.format(i_traj, name), b, 30))

    def _end_reset(self):
        if self._hp.wait_during_resetend:
            _ = raw_input("PRESS ENTER TO CONINUE")
    
        if self._hp.opencv_tracking:
            assert self._desig_pix is not None, "Designated pixels must be set (call get_obj_desig_goal)"
            track_desig = copy.deepcopy(self._desig_pix)
            [c.start_tracking(track_desig[i]) for i, c in enumerate(self._cameras)]

        self._reset_previous_qpos()
        self._init_dynamics()
        self._reset_counter += 1
        return self._get_obs(), None

    def _goto_closest_neutral(self):
        self._controller.move_to_neutral()
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
        if self._save_video:
            [c.reset_recording() for c in self._cameras]

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
            self._controller.move_to_neutral()
        else:
            self._controller.open_gripper(True)
            self._controller.move_to_neutral()

        if self._cleanup_rate > 0 and self._reset_counter % self._cleanup_rate == 0 and self._reset_counter > 0:
            self._controller.redistribute_objects()
            self._goto_closest_neutral()

        self._controller.move_to_neutral()
        self._controller.open_gripper(False)
        rospy.sleep(0.5)
        self._reset_previous_qpos()

        if self._hp.start_state:
            xyz = np.array(self._hp.start_state[:3]) * (self._high_bound[:3] - self._low_bound[:3]) + self._low_bound[:3]
            theta = self._hp.start_state[3]
            self._move_to_state(xyz, theta, 2.)
        else:
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
                logging.getLogger('robot_logger').error("DeSYNC!")
                raise Image_Exception
            time_stamps.append(stamp)
            cam_imgs.append(image)

        for index, i in enumerate(time_stamps[:-1]):
            for j in time_stamps[index + 1:]:
                if abs(i - j) > self._obs_tol:
                    logging.getLogger('robot_logger').error('DeSYNC!')
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

        if self._hp.reset_before_eval:
            self._controller.open_gripper(True)
            self._controller.move_to_neutral()

        final_pix = select_points(self.render(), ['front', 'left'], 'final',
                                 save_dir, clicks_per_desig=1, n_desig=ntasks)

        goal_pix = self.get_goal_pix(target_width)
        final_pix = pix_resize(final_pix, target_width, self._width)
        start_pix = pix_resize(self._start_pix, target_width, self._width)

        final_dist, start_dist = np.linalg.norm(final_pix - goal_pix), np.linalg.norm(start_pix - goal_pix)
        improvement = start_dist - final_dist
        logging.getLogger('robot_logger').info('final_dist: {}'.format(final_dist))
        logging.getLogger('robot_logger').info('start dist: {}'.format(start_dist))
        logging.getLogger('robot_logger').info('improvement: {}'.format(improvement))

        if self._hp.opencv_tracking:
            [c.end_tracking() for c in self._cameras]

        return {'final_dist': final_dist, 'start_dist': start_dist, 'improvement': improvement}

    def get_obj_desig_goal(self, save_dir, collect_goal_image=False, ntasks=1):
        raw_input("Robot in safe position? Hit enter when ready...")
        self._goto_closest_neutral()
        self._controller.open_gripper(True)

        if collect_goal_image:
            logging.getLogger('robot_logger').info("PLACE OBJECTS IN GOAL POSITION")
            raw_input("When ready to annotate GOAL images press enter...")
            goal_imgs = self.render()
            goal_pix = select_points(goal_imgs, ['front', 'left'], 'goal',
                                     save_dir, clicks_per_desig=1, n_desig=ntasks)

            raw_input("Robot in safe position? Hit enter when ready...")
            self._goto_closest_neutral()
            self._controller.open_gripper(True)

            logging.getLogger('robot_logger').info("PLACE OBJECTS IN START POSITION")
            raw_input("When ready to annotate START images press enter...")

            self._start_pix = select_points(self.render(), ['front', 'left'], 'desig',
                                     save_dir, clicks_per_desig=1, n_desig=ntasks)
            self._goal_pix = copy.deepcopy(goal_pix)
            self._desig_pix = copy.deepcopy(self._start_pix)

            return goal_imgs, goal_pix
        else:
            logging.getLogger('robot_logger').info("PLACE OBJECTS IN START POSITION")
            raw_input("When ready to annotate START images press enter...")

            self._start_pix, self._goal_pix = select_points(self.render(), ['front', 'left'], 'desig_goal',
                                     save_dir, n_desig=ntasks)
            self._desig_pix = copy.deepcopy(self._start_pix)
            return copy.deepcopy(self._goal_pix)

    def get_goal_pix(self, target_width):
        return pix_resize(self._goal_pix, target_width, self._width)