#!/usr/bin/env python

import argparse
import rospy

import socket
import intera_interface
import intera_external_devices
from intera_interface import CHECK_VERSION

import numpy as np
import socket
import pdb

TOLERANCE = 0.01
class RobotController(object):

    def __init__(self):
        """Initializes a controller for the robot"""

        print("Initializing node... ")
        rospy.init_node("sawyer_custom_controller")
        rospy.on_shutdown(self.clean_shutdown)

        self._rs = intera_interface.RobotEnable(CHECK_VERSION)
        init_state = self._rs.state().enabled
        print("Robot enabled...")

        self.limb = intera_interface.Limb("right")

        self.sawyer_gripper = False
        if self.sawyer_gripper:
            self.gripper = intera_interface.Gripper("right")

            self.gripper.calibrate()
            self.gripper.set_velocity(self.gripper.MAX_VELOCITY)  # "set 100% velocity"),
            self.gripper.open()

        self.joint_names = self.limb.joint_names()
        print("Done initializing controller.")


    def set_joint_delta(self, joint_name, delta):
        """Move a single joint by a delta"""
        current_position = self.limb.joint_angle(joint_name)
        self.set_joint(joint_name, current_position + delta)

    def set_joint(self, joint_name, pos):
        """Move a single joint to a target position"""
        joint_command = {joint_name: pos}
        self.limb.set_joint_positions(joint_command)

    def set_joints(self, command):
        """Move joints to commmand"""
        self.limb.move_to_joint_positions(command)

    def move_to_joints(self, joint_angles):
        cmd = dict(list(zip(self.joint_names, joint_angles)))
        self.set_joints(cmd)

    def set_joints_nonblocking(self, command):
        """Move joints to commmand, resending until reached"""
        for i in range(100000):
            self.limb.set_joint_positions(command)
            current = self.limb.joint_angles()
            if np.all(abs(distance_between_commands(current, command)) < TOLERANCE):
                rospy.loginfo("Reached target")
                break
        rospy.loginfo("Finished motion")

    def set_gripper(self, action):
        if self.has_gripper:
            if action == "close":
                self.gripper.close()
            elif action == "open":
                self.gripper.open()
            elif action == "calibrate":
                self.gripper.calibrate()

    def set_neutral(self, speed = .2):
        # using a custom handpicked neutral position
        # starting from j0 to j6:
        neutral_jointangles = [0.412271, -0.434908, -1.198768, 1.795462, 1.160788, 1.107675, 2.068076]
        cmd = dict(list(zip(self.joint_names, neutral_jointangles)))

        self.limb.set_joint_position_speed(speed)

        done = False
        while not done:
            try:
                self.set_joints(cmd)
            except:
                print('retrying set neutral...')

            done = True

        # self.limb.move_to_neutral()

    def clean_shutdown(self):
        print("\nExiting example.")
        # if not init_state:
        #     print("Disabling robot...")
            # rs.disable()


def distance_between_commands(j1, j2):
    a = []
    b = []
    for joint in j1:
        a.append(j1[joint])
        b.append(j2[joint])
    return np.array(a) - np.array(b)