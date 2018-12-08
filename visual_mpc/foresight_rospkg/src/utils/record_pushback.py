#!/usr/bin/env python
import rospy

import pickle
from intera_interface import CHECK_VERSION

import intera_interface
import argparse
import visual_mpc.foresight_rospkg as foresight_rospkg


class Pushback_Recorder(object):
    def __init__(self):
        """
        Records joint data to a file at a specified rate.
        rate: recording frequency in Hertz
        """

        print("Initializing node... ")
        rospy.init_node("pushback_recorder")

        parser = argparse.ArgumentParser()
        parser.add_argument('robot', type=str, help='robot name')


        args = parser.parse_args()
        self.robotname = args.robot

        self.file = '/'.join(str.split(foresight_rospkg.__file__, "/")[
                        :-1]) + '/src/utils/pushback_traj_{}.pkl'.format(self.robotname)

        self.rs = intera_interface.RobotEnable(CHECK_VERSION)
        self.init_state = self.rs.state().enabled

        self.limb = intera_interface.Limb("right")


        self._navigator = intera_interface.Navigator()
        self.start_callid = self._navigator.register_callback(self.start_recording, 'right_button_ok')
        # Navigator Rethink button press
        self.stop_callid = self._navigator.register_callback(self.stop_recording, 'right_button_square')

        self.control_rate = rospy.Rate(800)

        self.collect_active = False
        rospy.on_shutdown(self.clean_shutdown)
        self.joint_pos = []


        print('ready for recording!')
        rospy.spin()

    def stop_recording(self, data):
        if data < 0:
            return
        print('stopped recording')
        self.collect_active = False
        # self.playback()
        self.clean_shutdown()

    def start_recording(self, data):
        if data < 0:
            return
        print('started recording')
        self.collect_active = True
        self.joint_pos = []
        while(self.collect_active):
            self.control_rate.sleep()
            self.joint_pos.append(self.limb.joint_angles())

        with open(self.file, 'wb') as f:
            pickle.dump(self.joint_pos, f)

        print('saved file to ', self.file)

    def clean_shutdown(self):
        """
       Switches out of joint torque mode to exit cleanly
       """
        print("\nExiting example...")



if __name__ == '__main__':
    P = Pushback_Recorder()  # playback file

