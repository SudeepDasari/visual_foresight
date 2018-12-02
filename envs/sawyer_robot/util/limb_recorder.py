import rospy
import numpy as np
from intera_core_msgs.msg import EndpointState
from threading import Condition


class LatestEEObs:
    def __init__(self):
        self._cv = Condition()
        self._latest_eep = None
        self._eep_sub = rospy.Subscriber('/robot/limb/right/endpoint_state', EndpointState, self._state_listener)

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


class LimbRecorder:
    def __init__(self, control_limb):
        self._limb = control_limb
        self._ep_handler = LatestEEObs()

    def get_state(self):
        joint_angles = self.get_joint_angles()
        joint_velocities = self.get_joint_angles_velocity()
        eep = self.get_endeffector_pose()

        return joint_angles, joint_velocities, eep

    def get_joint_names(self):
        return self._limb.joint_names()

    def get_joint_cmd(self):
        return self._limb.joint_angles()

    def get_joint_angles(self):
        return np.array([self._limb.joint_angle(j) for j in self._limb.joint_names()])

    def get_joint_angles_velocity(self):
        return np.array([self._limb.joint_velocity(j) for j in self._limb.joint_names()])

    def get_endeffector_pose(self):
        """
        Relies on /robot/limb/right/endpoint_state which updates at 100HZ
        Not suitable for control faster than 100 HZ
        :return: Current end point of gripper
        """
        return self._ep_handler.get_eep()

    def get_xyz_quat(self):
        eep = self.get_endeffector_pose()
        return eep[:3], eep[3:]


class LimbWSGRecorder(LimbRecorder):
    def __init__(self, wsg_controller):
        self._ctrl = wsg_controller
        LimbRecorder.__init__(self, wsg_controller.limb)

    def get_gripper_state(self):
        g_width, g_force = self._ctrl.get_gripper_status(integrate_force=True)
        close_thresh, open_thresh = self._ctrl.get_limits()

        gripper_status = (g_width - close_thresh) / (open_thresh - close_thresh)  #t = 1 --> open, and t = 0 --> closed

        return np.array([gripper_status]), np.array([g_force])

    def get_state(self):
        gripper_state, force_sensor = self.get_gripper_state()
        j_angles, j_vel, eep = LimbRecorder.get_state(self)
        return j_angles, j_vel, eep, gripper_state, force_sensor
