#!/usr/bin/env python
import rospy
import sys
import logging
if sys.version_info[0] < 3:
    import cPickle as pkl
else:
    import pickle as pkl
import argparse


class Pushback_Recorder(object):
    def __init__(self, robot_type, file_name):
        """
        Records joint data to a file at a specified rate.
        rate: recording frequency in Hertz
        """
        if robot_type == 'sawyer':
            import intera_interface
            from visual_mpc.envs.robot_envs.sawyer.sawyer_impedance import SawyerImpedanceController
            self._controller = SawyerImpedanceController('recorder_bot', False, gripper_attached='none')
            self._controller.move_to_neutral()

            # Navigator Rethink button press
            self._navigator = intera_interface.Navigator()
            self.start_callid = self._navigator.register_callback(self.start_recording, 'right_button_ok')
            self.stop_callid = self._navigator.register_callback(self.stop_recording, 'right_button_square')

        elif robot_type == 'baxter':
            from pynput import mouse
            from pynput import keyboard
            import baxter_interface
            from visual_mpc.envs.robot_envs.baxter.baxter_impedance import BaxterImpedanceController
            self._controller = BaxterImpedanceController('baxter', False, gripper_attached='none',limb = 'right')
            self._controller.move_to_neutral()
            # Navigator Rethink button press
            self._navigator = baxter_interface.Navigator('right')
            self.start_callid = self._navigator.button0_changed.connect(self.start_recording)
            self.stop_callid = self._navigator.button1_changed.connect(self.stop_recording)

            # self.start_callid = self._navigator.register_callback(self.start_recording, 'right_button_ok')
            # self.stop_callid = self._navigator.register_callback(self.stop_recording, 'right_button_square')
        else:
            raise NotImplementedError
        
        self._control_rate = rospy.Rate(800)
        self._collect_active = False
        self._joint_pos = []
        self._file = file_name

        logging.getLogger('robot_logger').info('ready for recording!')
        rospy.spin()
        
    # def button0(self,key):
    #     self.val = 0
    #     print("button0")
    # def button1(self,x,y,button,pressed):
    #     while(True):
    #         print(self.val)

    def stop_recording(self, data):
        # print("something happened")
        if data < 0:
            return
        logging.getLogger('robot_logger').info('stopped recording')
        with open(self._file, 'wb') as f:
            pkl.dump(self._joint_pos, f)

        logging.getLogger('robot_logger').info('saved file to {}'.format(self._file))
        self._controller.clean_shutdown()

    def start_recording(self, data):

        # print("first thing happened")
        if data < 0:
            return
        logging.getLogger('robot_logger').info('Recorded angles')
        
        self._control_rate.sleep()
        self._joint_pos.append(self._controller.get_joint_angles())

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, help='type of robot', default='sawyer')
    parser.add_argument('--fname', type=str, help='name of saved pickle file', default='recording.pkl')
    args = parser.parse_args()

    Pushback_Recorder(args.robot, args.fname)  # playback file
