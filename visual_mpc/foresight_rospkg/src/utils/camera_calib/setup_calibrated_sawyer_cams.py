#!/usr/bin/python
import rospy
import numpy as np
import argparse
import intera_interface
import intera_external_devices
from intera_interface import CHECK_VERSION

from intera_core_msgs.srv import (
    SolvePositionFK,
    SolvePositionFKRequest,
)
from sensor_msgs.msg import JointState


class MarkerTracker:
    def __init__(self, valid_markers, track_window = 10):
        self.valid_markers = valid_markers
        self.track_window = track_window
        self.reset_collection()

    def reset_collection(self):
        self.position_track = {m: (np.zeros((self.track_window, 3)), 0) for m in self.valid_markers}

    def marker_update(self, m):
        # print("GOT {} in frame {}".format(marker_id, m.header.frame_id))
        marker_id = m.id
        marker_xyz = np.array([m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z])

        if marker_id in self.valid_markers:
            marker_history, marker_cnts = self.position_track[marker_id]
            self.position_track[marker_id] = (
            np.concatenate((marker_xyz.reshape(1, 3), marker_history[:-1, :])), marker_cnts + 1)

    def marker_callback(self, data):
        markers = data.markers

        for m in markers:
            self.marker_update(m)

    def print_marker_stats(self):
        for m in self.position_track.keys():
            if self.position_track[m][1] >= self.track_window:
                print("Marker, {} observed {} times".format(m, self.position_track[m][1]))
                print("mean pos {}".format(np.mean(self.position_track[m][0], axis = 0)))
            else:
                print("Marker, {} observed only {} times".format(m, self.position_track[m][1]))

    def print_calib_marker_stats(self, H_cam, t_cam):
        for m in self.position_track.keys():
            if self.position_track[m][1] >= self.track_window:
                print("Marker, {} observed {} times".format(m, self.position_track[m][1]))
                mean_pos = np.mean(self.position_track[m][0], axis = 0)
                print("mean pos {}".format(mean_pos))
                trans_pos = H_cam.dot(mean_pos.reshape((3, 1))) + t_cam.reshape((-1, 1))
                print("pred robot pos {}".format(trans_pos.reshape(-1)))
            else:
                print("Marker, {} observed only {} times".format(m, self.position_track[m][1]))


class CameraRegister:
    def __init__(self, cameras, valid_markers, limb, name_of_service, fksvc, validate = False):
        self.camera_trackers = {c : MarkerTracker(valid_markers) for c in cameras}
        self.collection = False
        self._limb_right = limb
        self.name_of_service = name_of_service
        self.fksvc = fksvc

        self.validate = validate
        if validate:
            if len(cameras) == 2:
                assert 0 in cameras, "FRONT CAMERA (0) is missing!"
                assert 1 in cameras, "LEFT CAMERA (1) is missing!"
                self.H_fcam = np.load('H_fcam.npy')
                self.t_fcam = np.load('t_fcam.npy')

                self.H_lcam = np.load('H_lcam.npy')
                self.t_lcam = np.load('t_lcam.npy')

                print("INITIALIZED IN VALIDATE_CALIB MODE")
            elif 'kinect2_rgb_optical_frame' in self.camera_trackers:
                self.H_fcam = np.load('H_kinect.npy')
                self.t_fcam = np.load('t_kinect.npy')
            else:
                raise NotImplementedError("Only handles 2 camera or kinect case")

    def toggle_collection(self, value):
        if not value:
            return

        if not self.collection:
            for c in self.camera_trackers.keys():
                self.camera_trackers[c].reset_collection()
            print("BEGINNING COLLECTION")
            self.collection = True
        else:
            print("ENDING COLLECTION")
            self.collection = False
            self.print_marker_stats()

    def print_callback(self, value):
        if value:
            self.print_marker_stats()

    def camera_track_callback(self, data):
        if not self.collection:
            return

        markers = data.markers
        for m in markers:
            cam_frameid = m.header.frame_id

            for c in self.camera_trackers.keys():
                if 'cam{}'.format(c) in cam_frameid:
                    self.camera_trackers[c].marker_update(m)
                elif c == 'kinect2_rgb_optical_frame' and 'kinect2_rgb_optical_frame' in cam_frameid:
                    self.camera_trackers[c].marker_update(m)

    def print_marker_stats(self):
        if self.validate and 'kinect2_rgb_optical_frame' in self.camera_trackers:
            self.camera_trackers['kinect2_rgb_optical_frame'].print_calib_marker_stats(self.H_fcam, self.t_fcam)
            return
        elif self.validate:
            print("FRONT CAMERA (assumed to be cam0)")
            self.camera_trackers[0].print_calib_marker_stats(self.H_fcam, self.t_fcam)

            print("SIDE CAMERA (assumed to be cam1)")
            self.camera_trackers[1].print_calib_marker_stats(self.H_lcam, self.t_lcam)
            return

        for c in self.camera_trackers.keys():
            print("CAMERA {}".format(c))
            self.camera_trackers[c].print_marker_stats()
            print ""

    def get_endeffector_pos(self):
        fkreq = SolvePositionFKRequest()
        joints = JointState()
        joints.name = self._limb_right.joint_names()
        joints.position = [self._limb_right.joint_angle(j)
                           for j in joints.name]

        # Add desired pose for forward kinematics
        fkreq.configuration.append(joints)
        fkreq.tip_names.append('right_hand')
        try:
            rospy.wait_for_service(self.name_of_service, 5)
            resp = self.fksvc(fkreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False

        pos = np.array([resp.pose_stamp[0].pose.position.x,
                        resp.pose_stamp[0].pose.position.y,
                        resp.pose_stamp[0].pose.position.z,
                        resp.pose_stamp[0].pose.orientation.x,
                        resp.pose_stamp[0].pose.orientation.y,
                        resp.pose_stamp[0].pose.orientation.z,
                        resp.pose_stamp[0].pose.orientation.w])

        return pos

    def print_robot_eep(self, value):
        if not value:
            return
        robot_pos = self.get_endeffector_pos()
        print("ROBOT XYZ POS")
        print(robot_pos[:3])
        print ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--markers', nargs='+',  dest='markers', help='Valid Marker Points')
    parser.add_argument('-c', '--cameras', nargs='+', dest='cameras', help='Operating Cameras')
    parser.add_argument('--ar_tracker', action='store_true',  help='Whether or not to subscribe to AR tracking messages')
    parser.add_argument('--validate_calib', action='store_true',  help='Validate Calibration')

    args = parser.parse_args()
    marker_list = args.markers
    camera_list = args.cameras
    v_check = args.validate_calib


    marker_set = {0, 1, 2}
    if marker_list is not None:
        marker_set = set([int(m) for m in marker_list])
    camera_set = {0, 1}
    if camera_list is not None:
        def int_or_kinect(c):
            if c == 'kinect':
                return 'kinect2_rgb_optical_frame'
            return int(c)

        camera_set = set([int_or_kinect(c) for c in camera_list])

    rospy.init_node('get_ar_points')

    #initialize robot
    rs = intera_interface.RobotEnable(CHECK_VERSION)
    init_state = rs.state().enabled
    limb_right = intera_interface.Limb("right")
    name_of_service = "ExternalTools/right/PositionKinematicsNode/FKService"
    fksvc = rospy.ServiceProxy(name_of_service, SolvePositionFK)

    #initialize tracking code
    camera_tracker = CameraRegister(camera_set, marker_set, limb_right, name_of_service, fksvc, validate=v_check)

    #init robot ui
    navigator = intera_interface.Navigator()
    navigator.register_callback(camera_tracker.toggle_collection, 'right_button_ok')
    navigator.register_callback(camera_tracker.print_callback, 'right_button_square')
    navigator.register_callback(camera_tracker.print_robot_eep, 'right_button_show')

    if args.ar_tracker:
        from ar_track_alvar_msgs.msg import AlvarMarkers
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, camera_tracker.camera_track_callback)

    print("BEGINNING SPIN")
    rospy.spin()


if __name__ == '__main__':
    main()