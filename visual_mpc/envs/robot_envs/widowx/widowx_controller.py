from visual_mpc.envs.robot_envs import RobotController
import numpy as np
import rospy
import time
from replab_core.config import *
from replab_core.utils import *
from replab_core.controller import *
import traceback
import logging
from moveit_commander.exception import MoveItCommanderException
from pyquaternion import Quaternion
from geometry_msgs.msg import Quaternion as Quaternion_msg
from geometry_msgs.msg import Pose, Point


class WidowXController(RobotController):
    def __init__(self, robot_name, print_debug, email_cred_file='', log_file='', control_rate=800, gripper_attached='default'):
        super(WidowXController, self).__init__(robot_name, print_debug, email_cred_file, log_file, control_rate, gripper_attached)
        self._widow_x = WidowX(boundaries=True)        

    def _init_gripper(self, gripper_attached):
        assert gripper_attached == 'default', 'widow_x only supports its default gripper at the moment'

    def move_to_neutral(self, duration=2):
        self._widow_x.move_to_neutral()

    def move_to_eep(self, target_pose, duration=1.5):
        """
        :param target_pose: Cartesian pose (x,y,z, quat).
        :param duration: Total time trajectory will take before ending
        """
        x, y, z = target_pose[:3]
        quat = Quaternion(target_pose[3:])
        quat_msg = Quaternion_msg(x = target_pose[4], y=target_pose[5], z=target_pose[6], w=target_pose[3]) 
        
        x = (x - CONTROL_NOISE_COEFFICIENT_BETA) / CONTROL_NOISE_COEFFICIENT_ALPHA
        y = (y - CONTROL_NOISE_COEFFICIENT_BETA) / CONTROL_NOISE_COEFFICIENT_ALPHA
        
        current_p = self._widow_x.commander.get_current_pose().pose
        p1 = Pose(position=Point(x=x, y=y, z=z), orientation=quat_msg)
        plan, f = self._widow_x.commander.compute_cartesian_path([current_p, p1], 0.001, 0.0)

        joint_goal = list(plan.joint_trajectory.points[-1].positions)

        first_servo = joint_goal[0]

        # TODO: This probs won't work well for more general action spaces, but it works find if roll=pitch=0
        joint_goal[4] = (quat.yaw_pitch_roll[0] - first_servo) % np.pi
        if joint_goal[4] > np.pi / 2:
            joint_goal[4] -= np.pi
        elif joint_goal[4] < -(np.pi / 2):
            joint_goal[4] += np.pi

        try:
            plan = self._widow_x.commander.plan(joint_goal)
        except MoveItCommanderException as e:
            logging.getLogger('robot_logger').error('Exception while planning')
            traceback.print_exc(e)
            self.clean_shutdown()
        
        self._widow_x.commander.execute(plan, wait=True)

    def move_to_ja(self, waypoints, duration=1.5):
        """
        :param waypoints: List of joint angle arrays. If len(waypoints) == 1: then go directly to point.
                                                      Otherwise: take trajectory that ends at waypoints[-1] and passes through each intermediate waypoint
        :param duration: Total time trajectory will take before ending
        """
        for waypoint in waypoints:
            # Move to each waypoint in sequence. See move_to_target_joint_position 
            # in controller.py 
            self._widow_x.move_to_target_joint_position(waypoint)

    def redistribute_objects(self):
        """
        Play pre-recorded trajectory that sweeps objects into center of bin
        """
        self._widow_x.sweep_arena()        

    def get_joint_angles(self):
        #returns current joint angles
        return np.array(self._widow_x.get_joint_values())

    def get_joint_angles_velocity(self):
        #returns current joint angles
        #No velocities for widowx? :(
        return None

    def get_cartesian_pose(self):
        #Returns cartesian end-effector pose
        current_pose = self._widow_x.get_current_pose().pose
        position = [current_pose.position.z, current_pose.position.y, current_pose.position.z]
        ori = current_pose.orientation
        quat = [ori.w, ori.x, ori.y, ori.z]
        return np.array(position + quat)
    
    def quat_2_euler(self, quat):
        return Quaternion(quat).yaw_pitch_roll

    def euler_2_quat(self, yaw=0.0, pitch=0.0, roll=0.0):
        yaw_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],[np.sin(yaw), np.cos(yaw), 0.0], [0, 0, 1.0]])
        pitch_matrix = np.array([[np.cos(pitch), 0., np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0, np.cos(pitch)]])
        roll_matrix = np.array([[1.0, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        rot_mat = yaw_matrix.dot(pitch_matrix.dot(roll_matrix))
        return Quaternion(matrix=rot_mat).elements

    def get_gripper_state(self, integrate_force=False):                         
        # returns gripper joint angle, force reading (none if no force)
        joint_angle = self._widow_x.gripper.get_current_joint_values()[0]
        return joint_angle, None

    def open_gripper(self, wait = False):                                       # should likely wrap separate gripper control class for max re-usability
        return self._widow_x.open_gripper()

    def close_gripper(self, wait = False):                                      # should likely wrap separate gripper control class for max re-usability
        return self._widow_x.close_gripper()

    @property
    def GRIPPER_CLOSE(self):
        return GRIPPER_CLOSE[0]
    
    @property
    def GRIPPER_OPEN(self):
        return GRIPPER_OPEN[1]
