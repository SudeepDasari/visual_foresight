#!/usr/bin/python

# Copyright (c) 2013-2017, Rethink Robotics Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import argparse

import rospy
import xacro

from intera_core_msgs.msg import (
    URDFConfiguration,
)

def xacro_parse(filename):
    doc = xacro.parse(None, filename)
    xacro.process_doc(doc, in_order=True)
    return doc.toprettyxml(indent='  ')

def send_urdf(parent_link, root_joint, urdf_filename, duration):
    """
    Send the URDF Fragment located at the specified path.

    @param parent_link: parent link to attach the URDF fragment to
                        (usually <side>_hand)
    @param root_joint: root link of the URDF fragment (usually <side>_gripper_base)
    @param urdf_filename: path to the urdf XML file to load into xacro and send
    @param duration: duration to repeat sending the URDF to ensure it is received
    """
    msg = URDFConfiguration()
    # The updating the time parameter tells
    # the robot that this is a new configuration.
    # Only update the time when an updated internal
    # model is required. Do not continuously update
    # the time parameter.
    msg.time = rospy.Time.now()
    # link to attach this urdf to onboard the robot
    msg.link = parent_link
    # root linkage in your URDF Fragment
    msg.joint = root_joint
    msg.urdf = xacro_parse(urdf_filename)
    pub = rospy.Publisher('/robot/urdf', URDFConfiguration, queue_size=10)
    rate = rospy.Rate(5) # 5hz
    start = rospy.Time.now()
    rospy.loginfo('publishing urdf')
    while not rospy.is_shutdown():
        pub.publish(msg)
        rate.sleep()
        if (rospy.Time.now() - msg.time) > rospy.Duration(duration):
            break

def main():
    """RSDK URDF Fragment Example:
    This example shows a proof of concept for
    adding your URDF fragment to the robot's
    onboard URDF (which is currently in use).
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-f', '--file', metavar='PATH', required=True,
        help='Path to URDF file to send'
    )
    required.add_argument(
        '-l', '--link', required=False, default="right_hand", #parent
        help='URDF Link already to attach fragment to (usually <left/right>_hand)'
    )
    required.add_argument(
        '-j', '--joint', required=False, default="right_gripper_base",
        help='Root joint for fragment (usually <left/right>_gripper_base)'
    )
    parser.add_argument("-d", "--duration", type=lambda t:abs(float(t)),
            default=5.0, help="[in seconds] Duration to publish fragment")
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node('rsdk_configure_urdf', anonymous=True)

    if not os.access(args.file, os.R_OK):
        rospy.logerr("Cannot read file at '%s'" % (args.file,))
        return 1
    send_urdf(args.link, args.joint, args.file, args.duration)
    return 0


def comp_gripper():
    import visual_mpc.envs.robot_envs.grippers.weiss as weiss_pkg
    urdf_frag = '/'.join(
        str.split(weiss_pkg.__file__, '/')[:-1]) + '/wsg50_xml/wsg_50_mod.urdf'
    rospy.init_node('rsdk_configure_urdf', anonymous=True)
    if not os.access(urdf_frag, os.R_OK):
        rospy.logerr("Cannot read file at '%s'" % (urdf_frag))
        sys.exit(1)
    send_urdf('right_hand', 'gripper_base_link', urdf_frag, 1e6)


if __name__ == '__main__':
    # sys.exit(main())
    comp_gripper()
