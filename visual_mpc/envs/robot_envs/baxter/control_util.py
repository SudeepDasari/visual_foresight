from pyquaternion import Quaternion
import inverse_kinematics
from visual_mpc.envs.util.interpolation import QuinticSpline
import logging
from baxter_core_msgs.msg import EndpointState
from threading import Condition
import rospy
import numpy as np
from geometry_msgs.msg import Quaternion as Quaternion_msg


CONTROL_RATE = 800
CONTROL_PERIOD = 1. / CONTROL_RATE
INTERP_SKIP = 16
NEUTRAL_JOINT_ANGLES = np.array([-0.79728651, -0.61090785,  1.00935936,  1.80856335, -1.07647102, 1.01166033,  0.01073787])
NEUTRAL_JOINT_CMD = {k:a for k, a in zip(['right_j{}'.format(i) for i in range(7)], NEUTRAL_JOINT_ANGLES)}
N_JOINTS = 7
max_vel_mag = np.array([0.88, 0.678, 0.996, 0.996, 1.776, 1.776, 2.316])
max_accel_mag = np.array([3.5, 2.5, 5, 5, 5, 5, 5])
RESET_SKIP = 800


def precalculate_interpolation(p1, q1, p2, q2, duration, last_pos, start_cmd, joint_names):
    q1, q2 = [Quaternion(x) for x in [q1, q2]]
    spline = QuinticSpline(p1, p2, duration)
    num_queries = int(CONTROL_RATE * duration / INTERP_SKIP) + 1
    jas = []
    last_cmd = start_cmd
    for t in np.linspace(0., duration, num_queries):
        cart_pos = spline.get(t)[0][0]
        interp_quat = Quaternion.slerp(q1, q2, t / duration)
        interp_pose = state_to_pose(cart_pos[:3], interp_quat.elements)
        z_angle = interp_quat.yaw_pitch_roll[0] + np.pi
        try:

            interp_ja = pose_to_ja(interp_pose, last_cmd,
                                   debug_z=z_angle * 180 / np.pi, retry_on_fail=True)
            last_cmd = interp_ja
            interp_ja = np.array([interp_ja[j] for j in joint_names])
            jas.append(interp_ja)
            last_pos = interp_ja
        except (EnvironmentError,KeyError) as error:
            jas.append(last_pos)
            logging.getLogger('robot_logger').error('ignoring IK failure or dictionary key not present error')

    interp_ja = []
    for i in range(len(jas) - 1):
        interp_ja.append(jas[i].tolist())
        for j in range(1, INTERP_SKIP):
            t = float(j) / INTERP_SKIP
            interp_point = (1 - t) * jas[i] + t * jas[i + 1]
            interp_ja.append(interp_point.tolist())
    interp_ja.append(jas[-1].tolist())

    return interp_ja


def state_to_pose(xyz, quat):
    """
    :param xyz: desired pose xyz
    :param quat: quaternion around z angle in [w, x, y, z] format
    :return: stamped pose
    """
    quat = Quaternion_msg(
        w=quat[0],
        x=quat[1],
        y=quat[2],
        z=quat[3]
    )

    desired_pose = inverse_kinematics.get_pose_stamped(xyz[0],
                                                       xyz[1],
                                                       xyz[2],
                                                       quat)
    return desired_pose


def pose_to_ja(target_pose, start_joints, tolerate_ik_error=False, retry_on_fail = False, debug_z = None):
    try:
        return inverse_kinematics.get_joint_angles(target_pose, seed_cmd=start_joints,
                                                        use_advanced_options=True)
    except ValueError:
        if retry_on_fail:
            logging.getLogger('robot_logger').error('retyring zangle was: {}'.format(debug_z))

            return pose_to_ja(target_pose, NEUTRAL_JOINT_CMD)
        elif tolerate_ik_error:
            raise ValueError("IK failure")    # signals to agent it should reset
        else:
            logging.getLogger('robot_logger').error('zangle was {}'.format(debug_z))
            raise EnvironmentError("IK Failure")   # agent doesn't handle EnvironmentError


class LatestEEObs:
    def __init__(self, limb="right"):
        self._cv = Condition()
        self._latest_eep = None
        self._eep_sub = rospy.Subscriber('/robot/limb/{}/endpoint_state'.format(limb), EndpointState, self._state_listener)

    def _state_listener(self, state_msg):
        pose = state_msg.pose

        with self._cv:
            self._latest_eep = np.array([pose.position.x,
                            pose.position.y,
                            pose.position.z,
                            pose.orientation.w,
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z])
            self._cv.notify_all()

    def get_eep(self):
        with self._cv:
            self._cv.wait()
            ee = self._latest_eep
        return ee
