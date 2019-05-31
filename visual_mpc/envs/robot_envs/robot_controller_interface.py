import logging
import rospy
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email.mime.text import MIMEText
from email import Encoders
import smtplib
import os
import sys
if sys.version_info[0] < 3:
    input_fn = raw_input
else:
    input_fn = input


class RobotController(object):
    def __init__(self, robot_name, print_debug, email_cred_file='', log_file='', control_rate=800, gripper_attached='wsg-50'):
        self._robot_name = robot_name
        rospy.init_node("foresight_robot_controller")
        rospy.on_shutdown(self.clean_shutdown)
        self._control_rate = rospy.Rate(control_rate)

        log_level = logging.INFO
        if print_debug:
            log_level = logging.DEBUG
        
        if email_cred_file and not log_file:
            log_file = '{}_log.txt'.format(self._robot_name)
        
        if log_file and os.path.exists(log_file):
            if input_fn('Log file exists. Okay deleting? (y/n):') == 'y':
                os.remove(log_file)
            else:
                exit(1)

        self._log_file = log_file

        self._is_email_setup = bool(email_cred_file)
        if self._is_email_setup:
            self._email_credentials = json.load(open(email_cred_file, 'r'))
        
        if self._is_email_setup or log_file:
            logging.basicConfig(filename=self._log_file, level=log_level)
        else:
            logging.basicConfig(level=log_level)

    def _send_email(self, message, attachment=None, subject='Data Collection Update'):
        try:
            # loads credentials and receivers
            assert self._is_email_setup, "no credentials"
            address, password = self._credentials['address'], self._credentials['password']
            if 'gmail' in address:
                smtp_server = "smtp.gmail.com"
            else:
                raise NotImplementedError
            receivers = self._credentials['receivers']

            # constructs message
            msg = MIMEMultipart()
            msg['Subject'] = subject 
            msg['From'] = address
            msg['To'] = ', '.join(receivers)
            msg.attach(MIMEText(message))

            if attachment:
                attached_part = MIMEBase('application', "octet-stream")
                attached_part.set_payload(open(attachment, "rb").read())
                Encoders.encode_base64(attached_part)
                attached_part.add_header('Content-Disposition', 'attachment; filename="{}"'.format(attachment))
                msg.attach(attached_part)

            # logs in and sends
            server = smtplib.SMTP_SSL(smtp_server)
            server.login(address, password)
            server.sendmail(address, receivers, msg.as_string())
        except:
            logging.error('email failed! check credentials (either incorrect or not supplied)')

    def clean_shutdown(self):
        if self._is_email_setup and self._log_file:
            self._send_email("Collection on {} has exited!".format(self._robot_name), attachment=self._log_file)
        elif self._is_email_setup:
            self._send_email("Collection on {} has exited!".format(self._robot_name))
        
        pid = os.getpid()
        print('Exiting example w/ pid: {}'.format(pid))
        os.kill(-pid, 9)

    def move_to_neutral(self, duration=2):
        raise NotImplementedError

    def move_to_eep(self, target_pose, duration=1.5):
        """
        :param target_pose: Cartesian pose (x,y,z, quat).
        :param duration: Total time trajectory will take before ending
        """
        raise NotImplementedError

    def move_to_ja(self, waypoints, duration=1.5):
        """
        :param waypoints: List of joint angle arrays. If len(waypoints) == 1: then go directly to point.
                                                     Otherwise: take trajectory that ends at waypoints[-1] and passes through each intermediate waypoint
        :param duration: Total time trajectory will take before ending
        """
        raise NotImplementedError

    def redistribute_objects(self):
        """
        Play pre-recorded trajectory that sweeps objects into center of bin
        """
        raise NotImplementedError

    def quat_2_euler(self, quat):
        # calculates and returns: yaw, pitch, roll from given quaternion
        raise NotImplementedError

    def euler_2_quat(self, yaw=0.0, pitch=0.0, roll=0.0):
        # calculates and returns quaternion from given euler angles
        raise NotImplementedError
    
    def get_state(self):
        # return joint_angles, joint_velocities, eep
        return self.get_joint_angles(), self.get_joint_angles_velocity(), self.get_cartesian_pose()

    def get_joint_angles(self):
        #returns current joint angles
        raise NotImplementedError

    def get_joint_angles_velocity(self):
        #returns current joint angle velocities
        raise NotImplementedError

    def get_cartesian_pose(self):
        #Returns cartesian end-effector pose
        raise NotImplementedError

    def get_xyz_quat(self):
        # separates cartesian pose into xyz, quaternion arrays
        eep = self.get_cartesian_pose()
        return eep[:3], eep[3:]

    def get_gripper_state(self, integrate_force=False):                         # should likely wrap separate gripper control class for max re-usability
        # returns gripper joint angle, force reading (none if no force)
        raise NotImplementedError

    def get_gripper_limits(self):                                               # should likely wrap separate gripper control class for max re-usability
        return self.GRIPPER_CLOSE, self.GRIPPER_OPEN

    def open_gripper(self, wait = False):                                       # should likely wrap separate gripper control class for max re-usability
        raise NotImplementedError

    def close_gripper(self, wait = False):                                      # should likely wrap separate gripper control class for max re-usability
        raise NotImplementedError

    @property
    def GRIPPER_CLOSE(self):
        raise NotImplementedError
    
    @property
    def GRIPPER_OPEN(self):
        raise NotImplementedError

    def quat_2_euler(self, quat):
        # calculates and returns: yaw, pitch, roll from given quaternion
        raise NotImplementedError

    def euler_2_quat(self, yaw=0.0, pitch=0.0, roll=0.0):
        # converts yaw, pitch, roll to quaternion 
        raise NotImplementedError
