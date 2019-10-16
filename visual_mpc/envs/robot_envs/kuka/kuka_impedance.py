import rospy
from pyquaternion import Quaternion
from visual_mpc.envs.robot_envs import RobotController
import numpy as np
from visual_mpc.envs.util.interpolation import CSpline
#from baxter_core_msgs.msg import JointCommand
import cPickle as pkl
import os
import logging
from .control_util import precalculate_interpolation, LatestEEObs, CONTROL_PERIOD, NEUTRAL_JOINT_ANGLES, NEUTRAL_JOINT_CMD, \
                          N_JOINTS, max_accel_mag, max_vel_mag, RESET_SKIP
import visual_mpc.envs.robot_envs as robot_envs

from visual_mpc.envs.robot_envs.kuka.kuka_interface import KukaInterface
from moveit_msgs.msg import RobotState, Constraints, OrientationConstraint



class KukaImpedanceController(RobotController):
    def __init__(self,
                 robot_name,
                 print_debug,
                 email_cred_file='',
                 log_file='',
                 control_rate=800,
                 gripper_attached='kuka_gripper'):

        super(KukaImpedanceController, self).__init__(robot_name,
                                                        print_debug,
                                                        email_cred_file,
                                                        log_file,
                                                        control_rate,
                                                        gripper_attached)
        
        # self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION) *********check the enabling of the robot 
        # self._limb_name = limb ##************ Do we have a defined limb 
        # self._limb = KukaInterface.Limb(self._limb_name) ###
        # #self._limb = baxter_interface.Limb(self._limb_name) ## *********** Create limb object
        # self.joint_names = self._limb.joint_names() 
        # self._ep_handler = LatestEEObs(limb=self._limb_name) ##### *** for getting position ****** ###

        ## .. check if this is publishing the joing angle commands 
        #self._cmd_publisher = rospy.Publisher('/robot/limb/{}/joint_command'.format(self._limb_name), JointCommand, queue_size=100)

        self.KukaObj = KukaInterface();
        self.joint_names = self.KukaObj.get_kuka_joint_angles_names()
    
    def _init_gripper(self, gripper_attached):
        if gripper_attached == 'none':
            from visual_mpc.envs.robot_envs import GripperInterface
            self._gripper = GripperInterface()
        elif gripper_attached == 'kuka_gripper':
            from visual_mpc.envs.robot_envs.grippers.kuka.default_kuka_gripper import KukaDefaultGripper
            self._gripper = KukaDefaultGripper()
        else:
            logging.getLogger('robot_logger').error("Gripper type '{}' not supported!".format(gripper_attached))
            raise NotImplementedError

    def _close_gripper_handler(self, value):
        pass
        # if value and self._gripper:
        #     midpoint = (self.GRIPPER_OPEN + self.GRIPPER_CLOSE) / 2.0
        #     gripper_width = self._gripper.get_gripper_state()[0]

        #     if gripper_width >= 40:
        #         self.close_gripper()    #close gripper on button release
        #     else:
        #         self.open_gripper()

    def _try_enable(self):
        """
        The start impedance script will try to re-enable the robot once it disables
        The script will wait until that occurs and throw an assertion if it doesn't
        """
        # i = 0
        # while not self._rs.state().enabled and i < 50:
        #     rospy.sleep(10)
        #     i += 1
        # if not self._rs.state().enabled:

        #*** Check how the kuka can be seen as enabled or not *************###
        if not self.KukaObj.Robot_State():
            logging.getLogger('robot_logger').error("Robot was disabled, please manually re-enable!")
            self.clean_shutdown()
    
    def move_to_neutral(self, duration=2):
        self.KukaObj.move_kuka_to_neutral()

    def move_to_eep(self, target_pose, duration=1.5):
        """
        :param target_pose: Cartesian pose (x,y,z, quat). 
        :param duration: Total time trajectory will take before ending
        """
        self._try_enable()


        # logging.getLogger('robot_logger').debug('Target Cartesian position in interface: {}'.format(target_pose))

        self.KukaObj.move_kuka_to_eep(target_pose)


        #p1, q1 = self.get_xyz_quat()
        #p2, q2 = target_pose[:3], target_pose[3:]

        #last_pos = self.get_joint_angles()
        #last_cmd = self._limb.joint_angles()
        #joint_names = self._limb.joint_names()

        #interp_jas = precalculate_interpolation(p1, q1, p2, q2, duration, last_pos, last_cmd, joint_names)

        #i = 0
        #self._control_rate.sleep()
        #start_time = rospy.get_time()
        #t = rospy.get_time()
        #while t - start_time < duration:
        #    lookup_index = min(int(min((t - start_time), duration) / CONTROL_PERIOD), len(interp_jas) - 1)
        #    self._send_pos_command(interp_jas[lookup_index])
        #    i += 1
        #    self._control_rate.sleep()
        #    t = rospy.get_time()
        #logging.getLogger('robot_logger').debug('Effective rate: {} Hz'.format(i / (rospy.get_time() - start_time)))   

#### ****************************************** Doubtful about this function. Still using position mode?  *******************************####

    def move_to_ja(self, waypoints, duration=1.5):
        """
        :param waypoints: List of joint angle arrays. If len(waypoints) == 1: then go directly to point.
                                                      Otherwise: take trajectory that ends at waypoints[-1] and passes through each intermediate waypoint
        :param duration: Total time trajectory will take before ending
        """
        self._try_enable()
        self.KukaObj.move_kuka_to_ja()

        # jointnames = self._limb.joint_names()
        # prev_joint = np.array([self._limb.joint_angle(j) for j in jointnames])
        # waypoints = np.array([prev_joint] + waypoints)

        # spline = CSpline(waypoints, duration)

        # start_time = rospy.get_time()  # in seconds
        # finish_time = start_time + duration  # in seconds

        # time = rospy.get_time()
        # while time < finish_time:
        #     pos, velocity, acceleration = spline.get(time - start_time)
        #     command = JointCommand()
        #     command.mode = JointCommand.POSITION_MODE
        #     command.names = jointnames
        #     command.command = pos
        #     #command.velocity = np.clip(velocity, -max_vel_mag, max_vel_mag)
        #     #command.acceleration = np.clip(acceleration, -max_accel_mag, max_accel_mag)
        #     self._cmd_publisher.publish(command)

        #     self._control_rate.sleep()
        #     time = rospy.get_time()

        # for i in xrange(10):
        #     command = JointCommand()
        #     command.mode = JointCommand.POSITION_MODE
        #     command.names = jointnames
        #     #command.position = waypoints[-1]
        #     command.command = waypoints[-1]
        #     self._cmd_publisher.publish(command)

        #     self._control_rate.sleep()

##### ********************************************************************************************#####

    def _send_pos_command(self, pos):
        self._try_enable()

        command = JointCommand()
        command.mode = JointCommand.POSITION_MODE
        command.names = self._limb.joint_names()
        #command.position = pos
        command.command = pos
        self._cmd_publisher.publish(command)

    def redistribute_objects(self):
        """
        Play pre-recorded trajectory that sweeps objects into center of bin
        """

        pass 
        # logging.getLogger('robot_logger').info('redistribute...')

        # file = '/'.join(str.split(robot_envs.__file__, "/")[
        #                 :-1]) + '/recorded_trajectories/pushback_traj_{}.pkl'.format(self._robot_name)

        # joint_pos = pkl.load(open(file, "rb"))

        # for t in range(0, len(joint_pos), RESET_SKIP):
        #     joint_t = joint_pos[t]
        #     if isinstance(joint_t, np.ndarray):
        #         pos_arr = joint_t
        #     else:
        #         pos_arr = np.array([joint_pos[t][j] for j in self._limb.joint_names()])
            
        #     self.move_to_ja([pos_arr])

    def get_joint_angles(self):
        # returns current joint angles
        return self.KukaObj.get_kuka_joint_angles()


        ####### *********************** Call MOVE IT Wrapper Class and get the joint angles *******###
        

        # return np.array([self._limb.joint_angle(j) for j in self._limb.joint_names()])

    def get_joint_angles_velocity(self):

        return self.KukaObj.get_kuka_joint_angles_velocity()

        ### ***************** Call MOVE IT wrapper class and get the joing angle velocities **************##
        

        #returns current joint angle velocities
        # return np.array([self._limb.joint_velocity(j) for j in self._limb.joint_names()])

    def get_cartesian_pose(self):

        eep = self.KukaObj.get_kuka_cartesian_pose()


        # logging.getLogger('robot_logger').debug('current Cartesian position in interface: {}'.format(eep))


        return eep         #**** CALL MOVE IT WRAPPER CLASS AND GET THE CURRENT POSITION ************#####

        #Returns cartesian end-effector pose
        # return self._ep_handler.get_eep()

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
