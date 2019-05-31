import rospy
from pyquaternion import Quaternion
from visual_mpc.envs.robot_envs import RobotController
import numpy as np
from visual_mpc.envs.util.interpolation import CSpline
from intera_core_msgs.msg import JointCommand
import cPickle as pkl
import intera_interface
import os
from visual_mpc.envs.robot_envs import RobotController
import logging
from .control_util import precalculate_interpolation, LatestEEObs, CONTROL_PERIOD
import visual_mpc.envs.robot_envs as robot_envs


NEUTRAL_JOINT_ANGLES = np.array([0.412271, -0.434908, -1.198768, 1.795462, 1.160788, 1.107675, -1.11748145])
NEUTRAL_JOINT_CMD = {k:a for k, a in zip(['right_j{}'.format(i) for i in range(7)], NEUTRAL_JOINT_ANGLES)}
N_JOINTS = 7
max_vel_mag = np.array([0.88, 0.678, 0.996, 0.996, 1.776, 1.776, 2.316])
max_accel_mag = np.array([3.5, 2.5, 5, 5, 5, 5, 5])
RESET_SKIP = 800


class SawyerImpedanceController(RobotController):
    def __init__(self, robot_name, print_debug, email_cred_file='', log_file='', control_rate=800, gripper_attached='wsg-50'):
        super(SawyerImpedanceController, self).__init__(robot_name, print_debug, email_cred_file, log_file, control_rate, gripper_attached)
        self._rs = intera_interface.RobotEnable(intera_interface.CHECK_VERSION)
        self._limb = intera_interface.Limb("right")
        self.joint_names = self._limb.joint_names()
        self._ep_handler = LatestEEObs()
        self._cmd_publisher = rospy.Publisher('/robot/limb/right/joint_command', JointCommand, queue_size=100)

        if gripper_attached == 'none':
            self._gripper = None
        elif gripper_attached == 'wsg-50':
            from visual_mpc.envs.robot_envs.grippers.weiss.wsg50_gripper import WSG50Gripper
            self._gripper = WSG50Gripper()
        # elif gripper_attached == 'default':
        #     self._gripper = # init default sawyer gripper
        else:
            logging.getLogger('robot_logger').error("Gripper not supported!")
            raise NotImplementedError

    def _close_gripper_handler(self, value):
        if value and self._gripper:
            midpoint = (self.GRIPPER_OPEN + self.GRIPPER_CLOSE) / 2.0
            gripper_width = self._gripper.get_gripper_state()[0]

            if gripper_width >= 40:
                self.close_gripper()    #close gripper on button release
            else:
                self.open_gripper()

    def _try_enable(self):
        """
        The start impedance script will try to re-enable the robot once it disables
        The script will wait until that occurs and throw an assertion if it doesn't
        """
        i = 0
        while not self._rs.state().enabled and i < 50:
            rospy.sleep(10)
            i += 1
        if not self._rs.state().enabled:
            logging.getLogger('robot_logger').error("Robot was disabled, please manually re-enable!")
            self.clean_shutdown()
    
    def move_to_neutral(self, duration=2):
        waypoints = [NEUTRAL_JOINT_ANGLES]
        self.move_to_ja(waypoints, duration)

    def move_to_eep(self, target_pose, duration=1.5):
        """
        :param target_pose: Cartesian pose (x,y,z, quat). 
        :param duration: Total time trajectory will take before ending
        """
        p1, q1 = self.get_xyz_quat()
        p2, q2 = target_pose[:3], target_pose[3:]

        last_pos = self.get_joint_angles()
        last_cmd = self._limb.joint_angles()
        joint_names = self._limb.joint_names()

        interp_jas = precalculate_interpolation(p1, q1, p2, q2, duration, last_pos, last_cmd, joint_names)

        i = 0
        self._control_rate.sleep()
        start_time = rospy.get_time()
        t = rospy.get_time()
        while t - start_time < duration:
            lookup_index = min(int(min((t - start_time), duration) / CONTROL_PERIOD), len(interp_jas) - 1)
            self._send_pos_command(interp_jas[lookup_index])
            i += 1
            self._control_rate.sleep()
            t = rospy.get_time()
        logging.getLogger('robot_logger').debug('Effective rate: {} Hz'.format(i / (rospy.get_time() - start_time)))   

    def move_to_ja(self, waypoints, duration=1.5):
        """
        :param waypoints: List of joint angle arrays. If len(waypoints) == 1: then go directly to point.
                                                      Otherwise: take trajectory that ends at waypoints[-1] and passes through each intermediate waypoint
        :param duration: Total time trajectory will take before ending
        """
        self._try_enable()

        jointnames = self._limb.joint_names()
        prev_joint = np.array([self._limb.joint_angle(j) for j in jointnames])
        waypoints = np.array([prev_joint] + waypoints)

        spline = CSpline(waypoints, duration)

        start_time = rospy.get_time()  # in seconds
        finish_time = start_time + duration  # in seconds

        time = rospy.get_time()
        while time < finish_time:
            pos, velocity, acceleration = spline.get(time - start_time)
            command = JointCommand()
            command.mode = JointCommand.POSITION_MODE
            command.names = jointnames
            command.position = pos
            command.velocity = np.clip(velocity, -max_vel_mag, max_vel_mag)
            command.acceleration = np.clip(acceleration, -max_accel_mag, max_accel_mag)
            self._cmd_publisher.publish(command)

            self._control_rate.sleep()
            time = rospy.get_time()

        for i in xrange(10):
            command = JointCommand()
            command.mode = JointCommand.POSITION_MODE
            command.names = jointnames
            command.position = waypoints[-1]
            self._cmd_publisher.publish(command)

            self._control_rate.sleep()

    def _send_pos_command(self, pos):
        self._try_enable()

        command = JointCommand()
        command.mode = JointCommand.POSITION_MODE
        command.names = self._limb.joint_names()
        command.position = pos
        self._cmd_publisher.publish(command)

    def redistribute_objects(self):
        """
        Play pre-recorded trajectory that sweeps objects into center of bin
        """
        self._debug_print('redistribute...')

        file = '/'.join(str.split(robot_envs.__file__, "/")[
                        :-1]) + '/recorded_trajectories/pushback_traj_{}.pkl'.format(self.robot_name)

        self.joint_pos = pkl.load(open(file, "rb"))

        for t in range(0, len(self.joint_pos), RESET_SKIP):
            # print(self.joint_pos[t])
            # self.set_joints(self.joint_pos[t])
            pos_arr = np.array([self.joint_pos[t][j] for j in self.limb.joint_names()])
            self.move_with_impedance([pos_arr])

    def get_joint_angles(self):
        #returns current joint angles
        return np.array([self._limb.joint_angle(j) for j in self._limb.joint_names()])

    def get_joint_angles_velocity(self):
        #returns current joint angle velocities
        return np.array([self._limb.joint_velocity(j) for j in self._limb.joint_names()])

    def get_cartesian_pose(self):
        #Returns cartesian end-effector pose
        return self._ep_handler.get_eep()

    def get_gripper_state(self, integrate_force=False):                         # should likely wrap separate gripper control class for max re-usability
        # returns gripper joint angle, force reading (none if no force)
        if self._gripper is None:
            logging.getLogger('robot_logger').warning("Attempting to get non-existent gripper's state!")
            return 0.0, 0.0
        return self._gripper.get_gripper_state(integrate_force)

    def get_gripper_limits(self):                                               # should likely wrap separate gripper control class for max re-usability
        return self.GRIPPER_CLOSE, self.GRIPPER_OPEN

    def open_gripper(self, wait = False):                                       # should likely wrap separate gripper control class for max re-usability
        if self._gripper is None:
            logging.getLogger('robot_logger').warning('Calling open on non-existent gripper!')
            return
        return self._gripper.open_gripper(wait)

    def close_gripper(self, wait = False):                                      # should likely wrap separate gripper control class for max re-usability
        if self._gripper is None:
            logging.getLogger('robot_logger').warning('Calling open on non-existent gripper!')
            return
        return self._gripper.close_gripper(wait)

    @property
    def GRIPPER_CLOSE(self):
        if self._gripper:
            return self._gripper.GRIPPER_CLOSE
        return 0
    
    @property
    def GRIPPER_OPEN(self):
        if self._gripper:
            return self._gripper.GRIPPER_OPEN
        return 0

    def quat_2_euler(self, quat):
        # calculates and returns: yaw, pitch, roll from given quaternion
        if not isinstance(quat, Quaternion):
            quat = Quaternion(quat)
        yaw, pitch, roll = quat.yaw_pitch_roll
        return yaw + np.pi, pitch, roll

    def euler_2_quat(self, yaw=np.pi/2, pitch=0.0, roll=np.pi):
        yaw = np.pi - yaw

        yaw_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],[np.sin(yaw), np.cos(yaw), 0.0], [0, 0, 1.0]])
        pitch_matrix = np.array([[np.cos(pitch), 0., np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0, np.cos(pitch)]])
        roll_matrix = np.array([[1.0, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        rot_mat = yaw_matrix.dot(pitch_matrix.dot(roll_matrix))
        return Quaternion(matrix=rot_mat).elements
