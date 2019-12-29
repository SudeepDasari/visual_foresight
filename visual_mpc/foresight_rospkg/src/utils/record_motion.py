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
            self._control_rate = rospy.Rate(800)
            self.stop_callid = self._navigator.register_callback(self.stop_recording, 'right_button_square')
        elif robot_type == 'widowx':
            import threading
            from visual_mpc.envs.robot_envs.widowx.widowx_controller import WidowXController
            from keyboard.msg import Key
            self._controller = WidowXController('recorder_bot', False)
            self._control_rate = rospy.Rate(50)
            self._controller.move_to_neutral()

            def keyboard_listener(msg):
                if msg.code == 115:
                    rec_thread = threading.Thread(target=self.start_recording, args=(1,))
                    rec_thread.start()
                else:
                    self.stop_recording(1)
                
            rospy.Subscriber("/keyboard/keydown", Key, keyboard_listener)

        elif robot_type == 'baxter':
            from pynput import mouse
            from pynput import keyboard
            import baxter_interface
            from visual_mpc.envs.robot_envs.baxter.baxter_impedance import BaxterImpedanceController
            self._controller = BaxterImpedanceController('baxter', False, gripper_attached='none',limb = 'right')
            self._controller.move_to_neutral()
            # Navigator Rethink button press
            self._navigator = baxter_interface.Navigator('right')
            self._navigator1 = baxter_interface.Navigator('left')


            self.start_callid = self._navigator.button0_changed.connect(self.start_recording)
            self.stop_callid = self._navigator1.button0_changed.connect(self.stop_recording)

        else:
            raise NotImplementedError
        
        
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
        print("something happened")
        if data < 0:
            return
        self._collect_active = False
        logging.getLogger('robot_logger').info('stopped recording')
        

    def start_recording(self, data):

        self._collect_active = True

        print("first thing happened")
        if data < 0:
            return
        logging.getLogger('robot_logger').info('recording')
        
        while(self._collect_active):
            self._joint_pos.append(self._controller.get_joint_angles())
            self._control_rate.sleep()
        
        logging.getLogger('robot_logger').info('Saving {} joing angles'.format(len(self._joint_pos)))
        with open(self._file, 'wb') as f:
            for p in self._joint_pos:
                print(p)
            
            pkl.dump(self._joint_pos, f)

        logging.getLogger('robot_logger').info('saved file to {}'.format(self._file))
        self._controller.clean_shutdown()

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, help='type of robot', default='sawyer')
    parser.add_argument('--fname', type=str, help='name of saved pickle file', default='recording.pkl')
    args = parser.parse_args()

    Pushback_Recorder(args.robot, args.fname)  # playback file
