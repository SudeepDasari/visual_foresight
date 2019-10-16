import rospy
from pyquaternion import Quaternion
from visual_mpc.envs.robot_envs import RobotController
import numpy as np
from visual_mpc.envs.robot_envs import GripperInterface
import cPickle as pkl
import requests
import os
import logging
import subprocess
import time

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
            time.sleep(3)
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
        pass

    def recover(self):
        requests.post('http://172.16.0.1:5000/clearerr')


    def _try_enable(self):
        """
        The start impedance script will try to re-enable the robot once it disables
        The script will wait until that occurs and throw an assertion if it doesn't
        """
        pass

    def move_to_neutral(self, duration=2):
        i = 0
        if (self.trialcount % 50 == 0) and (self.trialcount > 0):
            self.redistribute_objects()

        self.recover()

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

        """
        self.recover()

        print("*"*50)
        print(target_pose[:3])
        el = self.quat_2_euler(target_pose[3:])
        print(el)

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
