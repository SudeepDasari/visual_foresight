from visual_mpc.envs import robot_envs
import cPickle  as pkl
from visual_mpc.envs.robot_envs import RobotController
import numpy as np
import rospy
import time
from arbotix_python.arbotix import ArbotiX
from replab_core.config import *
import traceback
import logging
from pyquaternion import Quaternion
from geometry_msgs.msg import Quaternion as Quaternion_msg
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from visual_mpc.agent import Environment_Exception
from threading import Lock
import pybullet as p
import time


SERVO_IDS = [1, 2, 3, 4, 5, 6]
N_JOINTS = 6
MAX_TORQUE_L = 14
TORQUE_LIMIT = 34
GRIPPER_WAIT = 1.5
CONTROL_TOL = 1e-3
MAX_FINAL_ERR = 1
MAX_ERRORS=20


class WidowXController(RobotController):
    def __init__(self, robot_name, print_debug, email_cred_file='', log_file='', control_rate=100, gripper_attached='default'):
        super(WidowXController, self).__init__(robot_name, print_debug, email_cred_file, log_file, control_rate, gripper_attached)
        self._redist_rate = rospy.Rate(50)

        self._arbotix = ArbotiX('/dev/ttyUSB0')
        assert self._arbotix.syncWrite(MAX_TORQUE_L, [[servo_id, 255, 0] for servo_id in SERVO_IDS]) != -1, "failure during servo configuring"
        assert self._arbotix.syncWrite(TORQUE_LIMIT, [[servo_id, 255, 0] for servo_id in SERVO_IDS]) != -1, "failure during servo configuring"

        self._joint_lock = Lock()
        self._angles, self._velocities = {}, {}
        rospy.Subscriber("/joint_states", JointState, self._joint_callback)
        time.sleep(1)        

        self._joint_pubs = [rospy.Publisher('/{}/command'.format(name), Float64, queue_size=1) for name in JOINT_NAMES[:-1]]
        self._gripper_pub = rospy.Publisher('/gripper_prismatic_joint/command', Float64, queue_size=1)

        p.connect(p.DIRECT)
        widow_x_urdf = '/'.join(__file__.split('/')[:-1]) + '/widowx/widowx.urdf'
        self._armID = p.loadURDF(widow_x_urdf, useFixedBase=True) 
        p.resetBasePositionAndOrientation(self._armID, [0, 0, 0], p.getQuaternionFromEuler([np.pi, np.pi, np.pi]))       
        self._n_errors = 0

    def _init_gripper(self, gripper_attached):
        assert gripper_attached == 'default', 'widow_x only supports its default gripper at the moment'

    def _joint_callback(self, msg):
        with self._joint_lock:
            for name, position, velocity in zip(msg.name, msg.position, msg.velocity):
                self._angles[name] = position
                self._velocities[name] = velocity
    
    def _move_to_target_joints(self, joint_values):
        for value, pub in zip(joint_values, self._joint_pubs):
            pub.publish(Float64(value))

    def move_to_neutral(self, duration=2):
        self._n_errors = 0
        self._lerp_joints(np.array(NEUTRAL_VALUES), duration)
        self._gripper_pub.publish(Float64(GRIPPER_DROP[0]))
        time.sleep(GRIPPER_WAIT)
    
    def _reset_pybullet(self): 
        for i, angle in enumerate(self.get_joint_angles()):
            p.resetJointState(self._armID, i, angle)

    def _lerp_joints(self, target_joint_pos, duration):
        start_t, start_joints = rospy.get_time(), self.get_joint_angles()
        self._control_rate.sleep()
        cur_joints = self.get_joint_angles()
        while rospy.get_time() - start_t < 1.2 * duration and not np.isclose(target_joint_pos[:5], cur_joints[:5], atol=CONTROL_TOL).all():
            t = min(1.0, (rospy.get_time() - start_t) / duration)
            target_joints = (1 - t) * start_joints[:5] + t * target_joint_pos[:5]
            self._move_to_target_joints(target_joints)

            self._control_rate.sleep()
            cur_joints = self.get_joint_angles()
        logging.getLogger('robot_logger').debug('Lerped for {} seconds'.format(rospy.get_time() - start_t))
        self._reset_pybullet()

        delta = np.linalg.norm(target_joints[:5] - cur_joints[:5])
        if delta > MAX_FINAL_ERR:
            assert self._arbotix.syncWrite(TORQUE_LIMIT, [[servo_id, 255, 0] for servo_id in SERVO_IDS]) != -1, "failure during servo configuring"
            self._n_errors += 1
        if self._n_errors > MAX_ERRORS:
            logging.getLogger('robot_logger').error('More than {} errors! WidowX probably crashed.'.format(MAX_ERRORS))
            raise Environment_Exception
        
        logging.getLogger('robot_logger').debug('Delta at end of lerp is {}'.format(delta))
        
    
    def move_to_eep(self, target_pose, duration=1.5):
        """
        :param target_pose: Cartesian pose (x,y,z, quat).
        :param duration: Total time trajectory will take before ending
        """
        target_joint_pos = self._calculate_ik(target_pose[:3], target_pose[3:])[0]
        target_joint_pos = np.clip(np.array(target_joint_pos)[:6], JOINT_MIN, JOINT_MAX)
        self._lerp_joints(target_joint_pos, duration)

    def move_to_ja(self, waypoints, duration=1.5):
        """
        :param waypoints: List of joint angle arrays. If len(waypoints) == 1: then go directly to point.
                                                      Otherwise: take trajectory that ends at waypoints[-1] and passes through each intermediate waypoint
        :param duration: Total time trajectory will take before ending
        """
        for waypoint in waypoints:
            # Move to each waypoint in sequence. See move_to_target_joint_position 
            # in controller.py 
            self._move_to_target_joints(waypoint)

    def redistribute_objects(self):
        """
        Play pre-recorded trajectory that sweeps objects into center of bin
        """
        logging.getLogger('robot_logger').info('redistribute...')
        file = '/'.join(str.split(robot_envs.__file__, "/")[
                        :-1]) + '/recorded_trajectories/pushback_traj_{}.pkl'.format(self._robot_name)
        joint_pos = pkl.load(open(file, "rb"))

        self._redist_rate.sleep()
        for joint_t in joint_pos:
            print(joint_t)
            self._move_to_target_joints(joint_t)
            self._redist_rate.sleep()

    def get_joint_angles(self):
        #returns current joint angles
        with self._joint_lock:
            joints_ret = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'gripper_revolute_joint']
            try:
                return np.array([self._angles[k] for k in joints_ret])
            except KeyError:
                return None

    def get_joint_angles_velocity(self):
        #returns current joint angles
        #No velocities for widowx? :(
        with self._joint_lock:
            joints_ret = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'gripper_revolute_joint']
            try:
                return np.array([self._velocities[k] for k in joints_ret])
            except KeyError:
                return None

    def get_cartesian_pose(self):
        #Returns cartesian end-effector pose
        self._reset_pybullet()
        position, quat = p.getLinkState(self._armID, 5, computeForwardKinematics=1)[4:6]
        return np.array(list(position) + list(quat))
    
    def quat_2_euler(self, quat):
        roll, pitch, yaw = Quaternion(quat).yaw_pitch_roll
        return np.pi - yaw, pitch, roll

    def euler_2_quat(self, yaw=0.0, pitch=np.pi, roll=0.0):
        roll_matrix = np.array([[np.cos(roll), -np.sin(roll), 0.0],[np.sin(roll), np.cos(roll), 0.0], [0, 0, 1.0]])
        pitch_matrix = np.array([[np.cos(pitch), 0., np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0, np.cos(pitch)]])
        yaw_matrix = np.array([[1.0, 0, 0], [0, np.cos(yaw), -np.sin(yaw)], [0, np.sin(yaw), np.cos(yaw)]])
        rot_mat = roll_matrix.dot(pitch_matrix.dot(yaw_matrix))
        return Quaternion(matrix=rot_mat).elements

    def get_gripper_state(self, integrate_force=False):                         
        # returns gripper joint angle, force reading (none if no force)
        with self._joint_lock:
            joint_angle = self._angles['gripper_prismatic_joint_1']
        return joint_angle, None

    def open_gripper(self, wait = False):                                       # should likely wrap separate gripper control class for max re-usability
        self._gripper_pub.publish(Float64(self.GRIPPER_OPEN))
        time.sleep(GRIPPER_WAIT)

    def close_gripper(self, wait = False):                                      # should likely wrap separate gripper control class for max re-usability
        self._gripper_pub.publish(Float64(self.GRIPPER_CLOSE))
        time.sleep(GRIPPER_WAIT)

    @property
    def GRIPPER_CLOSE(self):
        return GRIPPER_CLOSED[0]
    
    @property
    def GRIPPER_OPEN(self):
        return GRIPPER_OPEN[1]

    def _calculate_ik(self, targetPos, targetQuat, threshold=1e-5, maxIter=1000, nJoints=6):
        closeEnough = False
        iter_count = 0
        dist2 = None
        
        best_ret, best_dist = None, float('inf')

        while (not closeEnough and iter_count < maxIter):
            jointPoses = list(p.calculateInverseKinematics(self._armID, 5, targetPos, targetQuat, JOINT_MIN, JOINT_MAX))
            for i in range(nJoints):
                jointPoses[i] = max(min(jointPoses[i], JOINT_MAX[i]), JOINT_MIN[i])
                p.resetJointState(self._armID, i, jointPoses[i])
            
            ls = p.getLinkState(self._armID, 5, computeForwardKinematics=1)
            newPos, newQuat = ls[4], ls[5]
            dist2 = sum([(targetPos[i] - newPos[i]) ** 2 for i in range(3)])
            closeEnough = dist2 < threshold
            iter_count += 1
            
            if dist2 < best_dist:
                best_ret, best_dist = (jointPoses, newPos, newQuat), dist2
        
        return best_ret 


if __name__ == '__main__':
    import pdb; pdb.set_trace()
    controller = WidowXController('widowx', True)
    controller.move_to_neutral()
    controller.redistribute_objects()
    for _ in range(100):
        x, y, z, theta = np.random.uniform([-0.1, -0.1, 0.3, np.pi / 2], [0.1, 0.1, 0.45, 3 * np.pi / 2])
        controller.move_to_eep([x, y, z] + controller.euler_2_quat(theta).tolist())
        print('xyz', controller.get_cartesian_pose()[:3])
    
    controller.move_to_eep([0,0,0.4] + controller.euler_2_quat(np.pi).tolist())
    recorded_pos = controller.get_cartesian_pose()
    print('xyz', recorded_pos[:3])
    print('yaw pitch roll', [np.rad2deg(i) for i in controller.quat_2_euler(recorded_pos[3:])])
    controller.move_to_neutral()
