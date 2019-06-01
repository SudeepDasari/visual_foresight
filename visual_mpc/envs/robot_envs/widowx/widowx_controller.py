import numpy as np
import rospy
import time
from replab_core.config import *
from replab_core.utils import *
from replab_core.controller import *
import traceback

class WidowXController(RobotController):

    def __init__(self, robot_name, print_debug, email_cred_file='', log_file=''):
        super(WidowXController, self).__init__(robot_name, print_debug, email_cred_file, log_file)
        self.widowx = WidowX(boundaries=True)        

    def move_to_neutral(self, duration=2):
        self.widowx.move_to_neutral()

    def move_to_eep(self, waypoints, duration=1.5):
        """
        :param waypoints: List of cartesian poses (x,y,z, quat). If len(waypoints) == 1: then go directly to point.
        :param duration: Total time trajectory will take before ending
        """
        
        current_p = self.widowx.commander.get_current_pose().pose
        waypoint_poses = [current_p]
        for pose in waypoints:

            x, y, z, quat = pose 
            x = (x - CONTROL_NOISE_COEFFICIENT_BETA) / CONTROL_NOISE_COEFFICIENT_ALPHA
            y = (y - CONTROL_NOISE_COEFFICIENT_BETA) / CONTROL_NOISE_COEFFICIENT_ALPHA
            conditions = [
                x <= BOUNDS_LEFTWALL,
                x >= BOUNDS_RIGHTWALL,
                y <= BOUNDS_BACKWALL,
                y >= BOUNDS_FRONTWALL,
                z <= BOUNDS_FLOOR,
                z >= 0.15
            ]

            print("Target Position: %0.4f, %0.4f, %0.4f" % (x, y, z))
            if not all(conditions):
                raise Exception("Targeted position is out of bounds!")

            p1 = Pose(position=Point(x=x, y=y, z=z), orientation=quat)
            waypoint_poses.append(p1)

        try:
            plan, f = self.widowx.commander.compute_cartesian_path(
                waypoint_poses, 0.001, 0.0)
        except MoveItCommanderException as e:
            print('Exception while planning')
            traceback.print_exc(e)
            return False

        return self.widowx.commander.execute(plan, wait=True)

    def move_to_ja(self, waypoints, duration=1.5):
       """
       :param waypoints: List of joint angle arrays. If len(waypoints) == 1: then go directly to point.
                                                     Otherwise: take trajectory that ends at waypoints[-1] and passes through each intermediate waypoint
       :param duration: Total time trajectory will take before ending
       """
       for waypoint in waypoints:
            # Move to each waypoint in sequence. See move_to_target_joint_position 
            # in controller.py 
            self.widowx.move_to_target_joint_position(waypoint)

    def redistribute_objects(self):
       """
       Play pre-recorded trajectory that sweeps objects into center of bin
    """
        self.widowx.sweep_arena()        

    def get_joint_angles(self):
        #returns current joint angles
        return self.widowx.get_joint_values()

    def get_joint_angles_velocity(self):
        #returns current joint angles
        #No velocities for widowx? :(
        #raise NotImplementedError
        return None

    def get_cartesian_pose(self):
       #Returns cartesian end-effector pose
       current_pose = self.widowx.get_current_pose().pose
       return current_pose.position + current_pose.orientation

    # Maybe create a separate WidowX gripper class?
    def get_gripper_state(self, integrate_force=False):                         
       # returns gripper joint angle, force reading (none if no force)
        joint_angle = self.widowx.gripper.get_current_joint_values()[0]
        return joint_angle, None

    def get_gripper_limits(self):                                               
       return GRIPPER_CLOSED[0], GRIPPER_OPEN[0]

    def open_gripper(self, wait = False):                                       
        self.widowx.open_gripper()  

    def close_gripper(self, wait = False):                                      
        self.widowx.close_gripper() 
