#!/usr/bin/python

import rospy
import numpy as np
import cPickle as pkl
import os

import scipy.spatial


class CalibratedCamera:
    def __init__(self, robot_name, camera_name):
        self.robot_name = robot_name
        calib_base = __file__.split('/')[:-1]
        self._calib_folder = '/'.join(calib_base + [self.robot_name])

        if os.path.exists('{}/H_{}.npy'.format(self._calib_folder, camera_name)):
            self.H_fcam = np.load('{}/H_{}.npy'.format(self._calib_folder, camera_name))
            self.t_fcam = np.load('{}/t_{}.npy'.format(self._calib_folder, camera_name))

        self._p2w_dict = pkl.load(open('{}/{}_{}_point_to_world.pkl'.format(self._calib_folder,
                                                                            self.robot_name, camera_name), 'rb'))

        self._camera_points = np.array([self._p2w_dict['top_left'], self._p2w_dict['top_right'],
                                  self._p2w_dict['bot_left'], self._p2w_dict['bot_right']])

        self._robot_points = np.array([self._p2w_dict['robot_top_left'], self._p2w_dict['robot_top_right'],
                                  self._p2w_dict['robot_bot_left'], self._p2w_dict['robot_bot_right']])

        self._cam_tri = scipy.spatial.Delaunay(self._camera_points)

    def camera_to_robot(self, camera_coord, name = 'front'):
        assert name == 'front', "calibration for camera_to_object not performed for left cam"
        robot_coords = []

        targets = np.array([c for c in camera_coord])
        target_triangle = self._cam_tri.find_simplex(targets)
        for i, t in enumerate(target_triangle):
            b = self._cam_tri.transform[t, :2].dot((targets[i].reshape(1, 2) - self._cam_tri.transform[t, 2]).T).T
            bcoord = np.c_[b, 1 - b.sum(axis=1)]

            points_robot_space = self._robot_points[self._cam_tri.simplices[t]]
            robot_coords.append(np.sum(points_robot_space * bcoord.T, axis=0))
        return robot_coords

