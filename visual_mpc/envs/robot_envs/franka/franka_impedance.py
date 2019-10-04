import rospy
from pyquaternion import Quaternion
from visual_mpc.envs.robot_envs import RobotController
import numpy as np
import geometry_msgs.msg as geom_msg
import franka_gripper
from franka_msgs.msg import FrankaState
# from franka_gripper import MoveActionGoal
# from visual_mpc.envs.util.interpolation import CSpline
# from intera_core_msgs.msg import JointCommand
from franka_gripper.msg import GraspActionGoal, MoveActionGoal, StopActionGoal
from franka_control.msg import ErrorRecoveryActionGoal
from std_msgs.msg import String
from visual_mpc.envs.robot_envs import GripperInterface
from sensor_msgs.msg import JointState
import cPickle as pkl
import requests
# from actionlib_msgs.msg import GoalID
# import intera_interface
import os
# from visual_mpc.envs.robot_envs import RobotController
import logging
import subprocess
import time
# from .control_util import precalculate_interpolation, LatestEEObs, CONTROL_PERIOD, NEUTRAL_JOINT_ANGLES, NEUTRAL_JOINT_CMD, \
#                           N_JOINTS, max_accel_mag, max_vel_mag, RESET_SKIP
# import visual_mpc.envs.robot_envs as robot_envs

class FrankaHand(GripperInterface):
    def __init__(self):
        self.currpos = 0.1
        self.lastsent = time.time()
    
    def get_gripper_state(self, integrate_force=False):
        # returns gripper joint angle, force reading (none if no force)
        return self.currpos, None

    def set_gripper(self, position, wait=False):
        print("CALLED GRIPPER", position)
        now = time.time()
        if (position == 0.03) and (self.currpos != position):
            delta = now - self.lastsent
            time.sleep(max(0, 2 - delta))
            requests.post('http://172.16.0.1:5000/close')
            self.lastsent = time.time()
        elif (position == 0.1) and (self.currpos != position):
            delta = now - self.lastsent
            time.sleep(max(0, 2 - delta))
            requests.post('http://172.16.0.1:5000/open')
            self.lastsent = time.time()
        else:
            pass
        self.currpos = position


    @property
    def GRIPPER_CLOSE(self):
        return 0.03

    @property
    def GRIPPER_OPEN(self):
        return 0.1

    def set_gripper_speed(self, new_speed):
        pass



class FrankaImpedanceController(RobotController):
    def __init__(self, robot_name, print_debug, email_cred_file='', log_file='', control_rate=1000, gripper_attached='hand'):
        super(FrankaImpedanceController, self).__init__(robot_name, print_debug, email_cred_file, log_file, control_rate, gripper_attached)
        # self.eepub = rospy.Publisher('/equilibrium_pose',
        #     geom_msg.PoseStamped ,queue_size=10)
        # self.jssub = rospy.Subscriber('/joint_states',
        #     JointState ,self.setjoint)
        # self.errsub = rospy.Subscriber('/franka_state_controller/franka_states',
        #     FrankaState ,self.saveerr)
        # self.resetpub = rospy.Publisher('/franka_control/error_recovery/goal',
        #     ErrorRecoveryActionGoal ,queue_size=1)
        # self.gripperesetpub = rospy.Publisher('/franka_gripper/stop/goal', \
        #         StopActionGoal ,queue_size=1)
        self.currpos = [0.5, 0.0, 0.15, 0.0, 0.0, 1.0, 0.0]
        self.trialcount = 0
        self.jp = None
        self.jv = None
    
    def setjoint(self, data):
        self.jp = np.array(data.position)
        self.jv = np.array(data.velocity)

    def saveerr(self, data):
        print(data)
        assert(False)
    
    def _init_gripper(self, gripper_attached):
        print(gripper_attached)
        if gripper_attached == 'hand':
            self._gripper = FrankaHand()
        else:
            logging.getLogger('robot_logger').error("Gripper not supported!")
            raise NotImplementedError

    def _close_gripper_handler(self, value):
        print("AAAA")

    def recover(self):
        requests.post('http://172.16.0.1:5000/clearerr')

    # def recover_gripper(self):
    #     msg = StopActionGoal()
    #     for i in range(10):
    #         self.gripperesetpub.publish(msg)



    def _try_enable(self):
        """
        The start impedance script will try to re-enable the robot once it disables
        The script will wait until that occurs and throw an assertion if it doesn't
        """
        pass
        # i = 0
        # while not self._rs.state().enabled and i < 50:
        #     rospy.sleep(10)
        #     i += 1
        # if not self._rs.state().enabled:
        #     logging.getLogger('robot_logger').error("Robot was disabled, please manually re-enable!")
        #     self.clean_shutdown()
    
    def move_to_neutral(self, duration=2):
        i = 0
        if (self.trialcount % 50 == 0) and (self.trialcount > 0):
            self.redistribute_objects()

        # if (self.trialcount % 20 == 0) or (self.trialcount % 20 == 1):
        #     self._send_pos_command([0.5, 0.0, 0.20, 0.0, 0.0, 1.0, 0.0])
        #     time.sleep(5)
        #     print("TO NEUTRAL")
        #     requests.post('http://172.16.0.1:5000/stopimp')
        #     print("Stopping Impedence")
        #     requests.post('http://172.16.0.1:5000/jointreset')
        #     print("Joint being reset")
        #     requests.post('http://172.16.0.1:5000/startimp')
        #     print("Starting Impedence")
            
            # self._send_pos_command([0.2, 0.0, 0.50, 0.0, 0.0, 1.0, 0.0])
            # time.sleep(5)
            # self._send_pos_command([0.2, 0.0, 0.20, 0.0, 0.0, 1.0, 0.0])
            # time.sleep(5)
            # self._send_pos_command([0.3, 0.0, 0.20, 0.0, 0.0, 1.0, 0.0])
        self.recover()
        # self.recover_gripper()
        # self.redistribute_objects()

        self._control_rate.sleep()
        start_time = rospy.get_time()
        t = rospy.get_time()
        while t - start_time < duration:
            self._send_pos_command([0.5, 0.0, 0.10, 0.0, 0.0, 1.0, 0.0])
            i += 1
            self._control_rate.sleep()
            t = rospy.get_time()
        self.trialcount += 1

    def move_to_eep(self, target_pose, duration=1.5, interpolate=False):
        """
        :param target_pose: Cartesian pose (x,y,z, quat). 
        :param duration: Total time trajectory will take before ending

        # Check for errors

        """
        # print(self.euler_2_quat(-0.142, 1.142, -2.142))
        # assert(False)
        # print("*"*50)
        self.recover()

        print("*"*50)
        print(target_pose[:3])
        el = self.quat_2_euler(target_pose[3:])
        print(el)
        # print("*"*50)
        # print("*"*50)
        if interpolate:
            cp = np.array(self.currpos)
            tp = np.array(target_pose)
            duration = 5
        i = 0
        self._control_rate.sleep()
        start_time = rospy.get_time()
        self.currpos = target_pose
        t = rospy.get_time()
        while t - start_time < duration:
            if interpolate:
                p = ((1.0*(t - start_time) / duration) * (tp - cp)) + cp
                self._send_pos_command(p)
            else:
                self._send_pos_command(target_pose)
            i += 1
            self._control_rate.sleep()
            t = rospy.get_time()
        logging.getLogger('robot_logger').debug('Effective rate: {} Hz'.format(i / (rospy.get_time() - start_time)))   

    # def move_to_ja(self, waypoints, duration=1.5):
    #     """
    #     :param waypoints: List of joint angle arrays. If len(waypoints) == 1: then go directly to point.
    #                                                   Otherwise: take trajectory that ends at waypoints[-1] and passes through each intermediate waypoint
    #     :param duration: Total time trajectory will take before ending
    #     """
    #     self._try_enable()

    #     jointnames = self._limb.joint_names()
    #     prev_joint = np.array([self._limb.joint_angle(j) for j in jointnames])
    #     waypoints = np.array([prev_joint] + waypoints)

    #     spline = CSpline(waypoints, duration)

    #     start_time = rospy.get_time()  # in seconds
    #     finish_time = start_time + duration  # in seconds

    #     time = rospy.get_time()
    #     while time < finish_time:
    #         pos, velocity, acceleration = spline.get(time - start_time)
    #         command = JointCommand()
    #         command.mode = JointCommand.POSITION_MODE
    #         command.names = jointnames
    #         command.position = pos
    #         command.velocity = np.clip(velocity, -max_vel_mag, max_vel_mag)
    #         command.acceleration = np.clip(acceleration, -max_accel_mag, max_accel_mag)
    #         self._cmd_publisher.publish(command)

    #         self._control_rate.sleep()
    #         time = rospy.get_time()

    #     for i in xrange(10):
    #         command = JointCommand()
    #         command.mode = JointCommand.POSITION_MODE
    #         command.names = jointnames
    #         command.position = waypoints[-1]
    #         self._cmd_publisher.publish(command)

    #         self._control_rate.sleep()

    def _send_pos_command(self, pos):
        arr = np.array(pos).astype(np.float32)
        data = {"arr": arr.tolist()}
        requests.post('http://172.16.0.1:5000/pose', json=data)

    def redistribute_objects(self):
        """
        Play pre-recorded trajectory that sweeps objects into center of bin
        """
        logging.getLogger('robot_logger').info('redistribute...')
        self.move_to_eep([0.5, 0.0, 0.15, 0.0, 0.0, 1.0, 0.0], interpolate=True)
        self.move_to_eep([0.8, 0.2, 0.15, 0.0, 0.0, 1.0, 0.0], interpolate=True)
        self.move_to_eep([0.8, 0.2, 0.00, 0.0, 0.0, 1.0, 0.0], interpolate=True)
        self.move_to_eep([0.5, 0.0, 0.00, 0.0, 0.0, 1.0, 0.0], interpolate=True)
        
        self.move_to_eep([0.5, 0.0, 0.15, 0.0, 0.0, 1.0, 0.0], interpolate=True)
        self.move_to_eep([0.8, -0.2, 0.15, 0.0, 0.0, 1.0, 0.0], interpolate=True)
        self.move_to_eep([0.8, -0.2, 0.00, 0.0, 0.0, 1.0, 0.0], interpolate=True)
        self.move_to_eep([0.5, 0.0, 0.00, 0.0, 0.0, 1.0, 0.0], interpolate=True)
        
        self.move_to_eep([0.5, 0.0, 0.15, 0.0, 0.0, 1.0, 0.0], interpolate=True)
        self.move_to_eep([0.25, 0.2, 0.15, 0.0, 0.0, 1.0, 0.0], interpolate=True)
        self.move_to_eep([0.25, 0.2, 0.00, 0.0, 0.0, 1.0, 0.0], interpolate=True)
        self.move_to_eep([0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], interpolate=True)
        
        self.move_to_eep([0.5, 0.0, 0.15, 0.0, 0.0, 1.0, 0.0], interpolate=True)
        self.move_to_eep([0.25, -0.2, 0.15, 0.0, 0.0, 1.0, 0.0], interpolate=True)
        self.move_to_eep([0.25, -0.2, 0.00, 0.0, 0.0, 1.0, 0.0], interpolate=True)
        self.move_to_eep([0.5, 0.0, 0.00, 0.0, 0.0, 1.0, 0.0], interpolate=True)

        # self.move_to_eep([0.5, 0.0, 0.15, 0.0, 0.0, 1.0, 0.0])
        # self.move_to_eep([0.5, 0.0, 0.25, 0.0, 0.0, 1.0, 0.0])
        # self.move_to_eep([0.5, 0.0, 0.35, 0.0, 0.0, 1.0, 0.0])
        # self.move_to_eep([0.5, 0.0, 0.50, 0.0, 0.0, 1.0, 0.0])
        # self.move_to_eep([0.5, 0.0, 0.60, 0.0, 0.0, 1.0, 0.0])
        # self.move_to_eep([0.5, 0.0, 0.50, 0.0, 0.0, 1.0, 0.0])
        # self.move_to_eep([0.5, 0.0, 0.35, 0.0, 0.0, 1.0, 0.0])
        # self.move_to_eep([0.5, 0.0, 0.25, 0.0, 0.0, 1.0, 0.0])
        # self.move_to_eep([0.5, 0.0, 0.15, 0.0, 0.0, 1.0, 0.0])
        # self.move_to_eep([0.5, 0.0, 0.1, 0.0, 0.0, 1.0, 0.0])



    def get_joint_angles(self):
        #returns current joint angles

        return self.jp

    def get_joint_angles_velocity(self):
        #returns current joint angle velocities
        return self.jv

    def get_cartesian_pose(self):
        #Returns cartesian end-effector pose
        return self.currpos

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
