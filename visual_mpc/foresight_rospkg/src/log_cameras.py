#!/usr/bin/env python
import argparse
from visual_mpc.envs.sawyer_robot.util.camera_recorder import CameraRecorder
from visual_mpc.envs.sawyer_robot.util.topic_utils import IMTopic
from visual_mpc.envs.sawyer_robot.util.user_interface import select_points
import datetime
import rospy
import os
import cv2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='logs current camera views')
    parser.add_argument("save_dir", type=str, help="Where to save views/calibration")
    parser.add_argument("--calibrate", action='store_true', default=False,
                        help='whether or not to take calibration points')
    parser.add_argument("--topic_names", type=str, nargs='+',
                        default=['front', 'left', 'right_side', 'left_side', 'right'],
                        help="list of camera topic names")
    args = parser.parse_args()
    rospy.init_node('save_cam_points')

    images = []
    for i, name in enumerate(args.topic_names):
        topic = IMTopic('/{}/image_raw'.format(name))
        recorder = CameraRecorder(topic)
        images.append(recorder.get_image()[1][:, :, ::-1].copy())

    now = datetime.datetime.today()
    folder_name = '{}/clicks_{}_{}_{}_{}'.format(args.save_dir, now.year, now.day, now.hour, now.minute)
    os.makedirs(folder_name)

    if args.calibrate:
        select_points(images, args.topic_names, '', folder_name, clicks_per_desig=1, n_desig=4)
    else:
        [cv2.imwrite('{}/{}.jpg'.format(folder_name, cam), i[:, :, ::-1]) for i, cam in zip(images, args.topic_names)]
