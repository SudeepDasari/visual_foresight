#!/usr/bin/env python

# You must run `export ROS_NAMESPACE=/iiwa` in the terminal before running this script
import sys
import copy
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import numpy as np
from moveit_msgs.msg import RobotState, Constraints, OrientationConstraint
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from os.path import join
import os
import random

class KukaInterface ():

    def __init__(self):

        self.bridge = CvBridge()

        joint_state_topic = ['joint_states:=/iiwa/joint_states']

        moveit_commander.roscpp_initialize(joint_state_topic)
        rospy.Subscriber("/iiwa/joint_states", JointState, self.State_callback)

        # Instantiate a RobotCommander object.  This object is
        # an interface to the robot as a whole.
        self.robot = moveit_commander.RobotCommander()
        self.group = moveit_commander.MoveGroupCommander("manipulator")
        # rospy.sleep(2)
        # self.scene = moveit_commander.PlanningSceneInterface('/iiwa/move_group/monitored_planning_scene')
        # box_pose = PoseStamped()
        # box_pose.header.frame_id = "world"
        # box_pose.pose.position.x = 1.0
        # box_pose.pose.orientation.w = 1.0
        # self.scene.add_box("test", box_pose, size=(0.1, 0.2, 0.3))

        # while not rospy.is_shutdown():
        #     rospy.sleep(1.0)
        #     for k in self.scene.__dict__.keys():
        #         print(k, self.scene.__dict__[k])
        #     # print(self.scene)
        #     print(self.scene.get_known_object_names())
        #     print(self.scene.get_attached_objects())
        # exit()

        self.group.set_max_velocity_scaling_factor(0.05)
        self.group.set_max_acceleration_scaling_factor(0.05)
        current_pose = self.group.get_current_pose(end_effector_link='iiwa_link_ee').pose

        self._joint_efforts = 0
        self._joint_vel = 0
        self._joint_name = 0
        self._header = None


        pose = PoseStamped()
        self.upright_constraints = Constraints()
        self.upright_constraints.name = "upright"
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = self.group.get_planning_frame()
        orientation_constraint.link_name = self.group.get_end_effector_link()
        pose.pose.orientation.x = 1.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 0.0

        orientation_constraint.orientation = pose.pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = .7#3.0
        orientation_constraint.absolute_y_axis_tolerance = .7#3.0
        orientation_constraint.absolute_z_axis_tolerance = 3.1415
        #orientation_constraint.absolute_z_axis_tolerance = 3.14 #ignore this axis
        orientation_constraint.weight = 1

        self.upright_constraints.orientation_constraints.append(orientation_constraint)

        self.group.allow_replanning(True)
        self.group.allow_looking(True)

        workspace = [0.5,-0.3,0.15,0.7,0.2,0.25]
        # self.group.set_workspace(workspace)
        # self.group.set_path_constraints(self.upright_constraints)

        self.traj_num = -1
        self.im_num = 0
        self.MAX_PATH_LENGTH = 15


        


    def Robot_State(self):
        
        if len(self.group.get_current_joint_values())>0:
            return True
        else:
            return False
     
    
    def State_callback(self,data):
        self._joint_efforts = data.effort
        self._joint_vel = data.velocity
        self._joint_name = data.name
        self._header = data.header

    def _calc_plan_statistics(self, plan, print_stats=False):
        if len(plan.joint_trajectory.points) == 0:
            rospy.logerr("Plan is empty. No statistics will be calculated")
            return

        total_distances = [0] * len(plan.joint_trajectory.points[0].positions)
        max_distances = [0] * len(plan.joint_trajectory.points[0].positions)
        max_vels = [0] * len(plan.joint_trajectory.points[0].positions)
        max_accels = [0] * len(plan.joint_trajectory.points[0].positions)

        for i, point in enumerate(plan.joint_trajectory.points):

            # Ignore wrist joint
            for j in range(len(point.positions) - 1):
                max_vels[j] = max(max_vels[j], abs(point.velocities[j]))
                max_accels[j] = max(max_accels[j], abs(point.accelerations[j]))

                if i > 0:
                    diff = abs(point.positions[j] - plan.joint_trajectory.points[i-1].positions[j])
                    max_distances[j] = max(max_distances[j], diff)
                    total_distances[j] += diff

            if print_stats:
                if abs(point.positions[0]) > np.pi / 2:
                    rospy.logerr("joint 0 to pos %f", point.positions[0])

                print "Positions:", point.positions


        if print_stats:
            print "\n\n\n\n\n\n\n"

            print "Total_distances:", total_distances
            print "Total distance:", sum(total_distances)
            print "max distance:", max_distances
            print "max of max_distances:", max(max_distances)
            print "max_vels:", max_vels
            print "max of vels:", max(max_vels)
            print "max_accels:", max_accels
            print "max of max_accels:", max(max_accels)
            print "\n\n\n\n\n\n\n"

        if max(max_distances) > 0.1:
            rospy.logerr("Max distance: %f", max(max_distances))
        if sum(total_distances) > 1.5:
            rospy.logerr("total move: %f", sum(total_distances))


        return sum(total_distances)


    def _plan_to_position(self, position):
        pose = [position[0],
                position[1],
                position[2],
                np.pi,
                0.0,
                0.0]

        replan_count = 0
        self.group.set_pose_target(pose, end_effector_link='iiwa_link_ee')
        plan = self.group.plan()

        move_distance = self._calc_plan_statistics(plan)
        print("plan length is", len(plan.joint_trajectory.points) )


        while len(plan.joint_trajectory.points) > self.MAX_PATH_LENGTH:
            print("Replan after plan length:", len(plan.joint_trajectory.points))
            print("replanned", replan_count, "times")
            pose[5] = 2 * np.pi * random.random()
            self.group.set_pose_target(pose, end_effector_link='iiwa_link_ee')
            plan = self.group.plan()
            replan_count += 1

            # if replan_count > 20 and len(plan.joint_trajectory.points) < 20:
            #     rospy.logerr("Exiting with lower standards.  This make break")
            #     break

            move_distance = self._calc_plan_statistics(plan)

            if replan_count > 20:
                rospy.logerr("Planning failed.  Attempting to reset position")
                self.move_kuka_to_neutral()
                replan_count = 0



        self._calc_plan_statistics(plan, print_stats=True)

        return plan

    def move_kuka_to_neutral(self):
        plan = self._plan_to_position([0.6,-0.05,0.4])
        # NEUTRAL_POSE= [0.6,-0.05,0.4,3.14159, 0.0, 0.0]
        # current_pose = self.group.get_current_pose(end_effector_link='iiwa_link_ee').pose
        # # print(self.group.get_current_joint_values())

        # # self.group.set_position_target(NEUTRAL_POSE[:3], end_effector_link='iiwa_link_ee')
        # self.group.set_pose_target(NEUTRAL_POSE, end_effector_link='iiwa_link_ee')
        # plan = self.group.plan()

        # print("Plan length:", len(plan.joint_trajectory.points))

        # while len(plan.joint_trajectory.points) > 15:
        #     print("Trying new random orientation")
        #     print("Plan length:", len(plan.joint_trajectory.points))
        #     NEUTRAL_POSE = [NEUTRAL_POSE[0], NEUTRAL_POSE[1], NEUTRAL_POSE[2], NEUTRAL_POSE[3], NEUTRAL_POSE[4], 2 * np.pi * random.random()]
        #     self.group.set_pose_target(NEUTRAL_POSE, end_effector_link='iiwa_link_ee')
        #     plan = self.group.plan()


        # print(self.group.get_current_joint_values())
        print("plan length executed is", len(plan.joint_trajectory.points) )
        if not plan.joint_trajectory.points:
            print "[ERROR] No trajectory found"
        else:
            # print(self.group.get_joint_value_target())
            self.group.go(wait=True)

        self.traj_num = self.traj_num + 1 
        

    def move_kuka_to_eep(self, target_pose):
        p, q = target_pose[:3], target_pose[3:]

        if p[0]>0.68:
            p[0] = 0.68
        elif p[0]<0.52:
            p[0] = 0.52

        if p[1]>0.18:
            p[1] = 0.18
        elif p[1]<-0.28:
            p[1] = -0.28

        if p[2]>0.25:
            p[2] = 0.25
        elif p[2]<0.15:
            p[2] = 0.15


        plan = self._plan_to_position(p)
        # goal_pose = [p[0], p[1], p[2], 3.14159, 0.0, 2 * np.pi * random.random()]

        current_pose = self.group.get_current_pose(end_effector_link='iiwa_link_ee').pose.position

        # # print(current_pose.position)
        # # print(current_pose.orientation) 

        # self.group.set_pose_target(goal_pose, end_effector_link='iiwa_link_ee')
        # # self.group.set_position_target(goal_pose[:3], end_effector_link='iiwa_link_ee')

        # plan = self.group.plan()

        # print("Plan length:", len(plan.joint_trajectory.points))

        # while len(plan.joint_trajectory.points) > self.MAX_PATH_LENGTH:
        #     print("Trying new random orientation")
        #     print("Plan length:", len(plan.joint_trajectory.points))
        #     goal_pose = [p[0], p[1], p[2], 3.14159, 0.0, 2 * np.pi * random.random()]
        #     self.group.set_pose_target(goal_pose, end_effector_link='iiwa_link_ee')
        #     plan = self.group.plan()

        print("plan length executed is", len(plan.joint_trajectory.points) )

        if not plan.joint_trajectory.points:
            print "[ERROR] No trajectory found"
        else:
            self.group.go(wait=True)

        target_position =  np.asarray([p[0],p[1],p[2]])

        current_position = np.asarray([current_pose.x,current_pose.y,current_pose.z])

        # while(np.linalg.norm(current_position-target_position)>0.01):
        #         current_pose = self.group.get_current_pose(end_effector_link='iiwa_link_ee').pose.position
        #         current_position = np.asarray([current_pose.x,current_pose.y,current_pose.z])
        #         print("position difference is = ", np.sum(current_position-target_position))
        


        

    def move_kuka_to_ja(self):
        pass
        """
        :param waypoints: List of joint angle arrays. If len(waypoints) == 1: then go directly to point.
                                                     Otherwise: take trajectory that ends at waypoints[-1] and passes through each intermediate waypoint
        :param duration: Total time trajectory will take before ending
        """
        #*** Probably dont need this *** ###

    def redistribute_kuka_objects(self):

        P1 = [0.5,-0.05,0.2,3.14159, 0.0, 0.0]
        P2 = [0.6,-0.05,0.2,3.14159, 0.0, 0.0]
        P3 = [0.5,-0.3,0.4,3.14159, 0.0, 0.0]

        P4 = [0.5,-0.3,0.2,3.14159, 0.0, 0.0]
        P5 = [0.6,-0.15,0.2,3.14159, 0.0, 0.0]
        P6 = [0.6,-0.3,0.4,3.14159, 0.0, 0.0]

        P7 = [0.6,-0.3,0.2,3.14159, 0.0, 0.0]
        P8 = [0.6,-0.15,0.2,3.14159, 0.0, 0.0]
        P9 = [0.7,-0.3,0.4,3.14159, 0.0, 0.0]

        P10 = [0.7,-0.3,0.2,3.14159, 0.0, 0.0]
        P11 = [0.6,-0.15,0.2,3.14159, 0.0, 0.0]
        P12 = [0.7,-0.05,0.4,3.14159, 0.0, 0.0]

        P13 = [0.7,-0.05,0.2,3.14159, 0.0, 0.0]
        P14 = [0.6,-0.05,0.2,3.14159, 0.0, 0.0]
        P15 = [0.7,0.2,0.4,3.14159, 0.0, 0.0]

        P16 = [0.7,0.2,0.2,3.14159, 0.0, 0.0]
        P17 = [0.6,0.1,0.2,3.14159, 0.0, 0.0]
        P18 = [0.6,0.2,0.4,3.14159, 0.0, 0.0]


        P19 = [0.6,0.2,0.2,3.14159, 0.0, 0.0]
        P20 = [0.6,0.1,0.2,3.14159, 0.0, 0.0]
        P21 = [0.5,0.2,0.4,3.14159, 0.0, 0.0]


        P22 = [0.5,0.2,0.2,3.14159, 0.0, 0.0]
        P23 = [0.6,0.1,0.2,3.14159, 0.0, 0.0]
        Pn= [0.5,-0.05,0.4,3.14159, 0.0, 0.0]

        redist_traj = [Pn,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,P14,P15,P16,P17,P18,P19,P20,P21,P22,P23,Pn]

        for i in redist_traj:
            self.group.set_pose_target(i, end_effector_link='iiwa_link_ee')

            plan = self.group.plan()

            if not plan.joint_trajectory.points:
                print "[ERROR] No trajectory found"
            else:
                self.group.go(wait=True)

            current_pos = self.group.get_current_pose(end_effector_link='iiwa_link_ee').pose.position
            current_position = np.asarray([current_pos.x,current_pos.y,current_pos.z])
            target_position = np.asarray(i[0:3])
            # print(target_position)
            while(abs(np.sum(current_position-target_position))>0.01):
                counter = 0
                current_pos = self.group.get_current_pose(end_effector_link='iiwa_link_ee').pose.position
                current_position = np.asarray([current_pos.x,current_pos.y,current_pos.z])
                print("position difference is = ", np.sum(current_position-target_position))
                counter = counter + 1
                if counter>10000000:
                    return




        """
        Play pre-recorded trajectory that sweeps objects into center of bin
        """
        pass 
    
    def get_kuka_state(self):
        # return joint_angles, joint_velocities, eep
        return self.group.get_current_joint_values(),self.get_kuka_joint_angles_velocity(),self.get_kuka_cartesian_pose()

        # return self.get_kuka_joint_angles(), self.get_kuka_joint_angles_velocity(), self.get_kuka_cartesian_pose()

    def get_kuka_joint_angles(self):
        #returns current joint angles
        return self.group.get_current_joint_values()

    def get_kuka_joint_angles_velocity(self):
        #returns current joint angle velocities
        # rospy.sleep(0.01)
        return self._joint_vel

    def get_kuka_joint_angles_names(self):
        #returns current joint angle velocities
        # rospy.sleep(0.01)
        return self._joint_name

    def get_kuka_joint_angles_effort(self):
        #returns current joint angle velocities
        # rospy.sleep(0.01)
        return self._joint_efforts

    def get_kuka_cartesian_pose(self):
        #Returns cartesian end-effector pose
        pose =  self.group.get_current_pose(end_effector_link='iiwa_link_ee').pose
        eep = np.array([pose.position.x,
                            pose.position.y,
                            pose.position.z,
                            pose.orientation.w,
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z])

        return eep

    def get_xyz_quat(self):
        # separates cartesian pose into xyz, quaternion arrays
        position = self.get_kuka_cartesian_pose().position
        orient = self.get_kuka_cartesian_pose().orientation
        return position.x,position.y,position.z,orient.x,orient.y,orient.z,orient.w

    def save_images(self):


        # base_path = "/home/server/Desktop/saved_images/"
        # data = rospy.wait_for_message('/camera1/usb_cam/image_raw', Image)
        # print("wait_for_message stamp:", data.header.stamp)

        # try:
        #     cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        # except CvBridgeError as e:
        #     print(e)

        # print "Saved to: ", base_path+str(0)+".jpg"
        # cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB, cv_image)
        # cv2.imwrite(join(base_path, "frame{:06d}.jpg".format(0)), cv_image)#*255)
        

        ### ********************#####

        path = "/home/server/Desktop/traj_images/" +str(self.traj_num)
        folders = ['Image0','Image1','Image2']
        for folder in folders:
            base_path = os.path.join(path,folder)
            if not os.path.exists(base_path):
                os.makedirs(base_path)

        
        data0 = rospy.wait_for_message('/camera1/usb_cam/image_raw', Image)
        data1 = rospy.wait_for_message('/camera2/usb_cam/image_raw', Image)
        data2 = rospy.wait_for_message('/camera3/usb_cam/image_raw', Image)
        print("wait_for_message stamp of camera 1:", data0.header.stamp,"\n")
        print("wait_for_message stamp of camera 2:", data1.header.stamp,"\n")
        print("wait_for_message stamp of camera 3:", data2.header.stamp,"\n")

        try:
            cv_image0 = self.bridge.imgmsg_to_cv2(data0, "passthrough")
            cv_image1 = self.bridge.imgmsg_to_cv2(data1, "passthrough")
            cv_image2 = self.bridge.imgmsg_to_cv2(data2, "passthrough")
        except CvBridgeError as e:
            print(e)

        print "Saved to: ", path+str(self.traj_num)
        cv2.cvtColor(cv_image0, cv2.COLOR_BGR2RGB, cv_image0)
        cv2.cvtColor(cv_image1, cv2.COLOR_BGR2RGB, cv_image1)
        cv2.cvtColor(cv_image2, cv2.COLOR_BGR2RGB, cv_image2)
        cv2.imwrite(join(path,"Image0", "frame{:06d}.jpg".format(self.im_num)), cv_image0)#*255)
        cv2.imwrite(join(path,"Image1", "frame{:06d}.jpg".format(self.im_num)), cv_image1)#*255)
        cv2.imwrite(join(path,"Image2", "frame{:06d}.jpg".format(self.im_num)), cv_image2)#*255)


         


if __name__ == '__main__':
    rospy.init_node("standalone_robot_controller", anonymous=True)
    kuka_obj = KukaInterface()
    try:
        kuka_obj.move_kuka_to_neutral()
    except rospy.ROSInterruptException:
        pass
