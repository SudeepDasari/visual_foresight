#!/usr/bin/python
import argparse
import logging
import rospy
import numpy as np


def main_sawyer():
    # import intera_interface
    from visual_mpc.envs.robot_envs.sawyer.sawyer_impedance import SawyerImpedanceController
    controller = SawyerImpedanceController('sawyer', True, gripper_attached='none')       # doesn't initial gripper object even if gripper is attached

    def print_eep(value):
        if not value:
            return
        xyz, quat = controller.get_xyz_quat()
        yaw, roll, pitch = [np.rad2deg(x) for x in controller.quat_2_euler(quat)]
        logging.getLogger('robot_logger').info("XYZ IS: {}, ROTATION IS: yaw={} roll={} pitch={}".format(xyz, yaw, roll, pitch))
    
    navigator = intera_interface.Navigator()
    navigator.register_callback(print_eep, 'right_button_show')
    rospy.spin()



def main_franka():
    from visual_mpc.envs.robot_envs.franka.franka_impedance import FrankaImpedanceController
    controller = FrankaImpedanceController('franka', True, gripper_attached='hand')       # doesn't initial gripper object even if gripper is attached

    def print_eep(value):
        if not value:
            return
        xyz, quat = controller.get_xyz_quat()
        yaw, roll, pitch = [np.rad2deg(x) for x in controller.quat_2_euler(quat)]
        print(xyz)
        logging.getLogger('robot_logger').info("XYZ IS: {}, ROTATION IS: yaw={} roll={} pitch={}".format(xyz, yaw, roll, pitch))
    
    print_eep(1.0)
    rospy.spin()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gets robot end effector points from controller')
    parser.add_argument('--robot', type=str, default='sawyer', help='robot being used')
    args = parser.parse_args()

    if args.robot == 'sawyer':
        main_sawyer()
    elif args.robot == 'franka':
        main_franka()
    
    raise NotImplementedError
