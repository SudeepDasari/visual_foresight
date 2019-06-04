#!/usr/bin/env python

from std_msgs.msg import (
    UInt16,
    Header
)
from geometry_msgs.msg import (
    PoseStamped,
    PointStamped,
    Pose,
    Point,
    Quaternion,
)
import rospy

from sensor_msgs.msg import JointState

import socket
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

def get_joint_angles(pose, seed_cmd = None, use_advanced_options = False, limb="right"):
    name_of_service = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(name_of_service, SolvePositionIK)
    ikreq = SolvePositionIKRequest()
    
    # Add desired pose for inverse kinematics
    ikreq.pose_stamp.append(pose)
    # Request inverse kinematics from base to "right_hand" link
#    ikreq.tip_names.append('right_hand')

    seed_joints = None
    if use_advanced_options:
        # Optional Advanced IK parameters
        # The joint seed is where the IK position solver starts its optimization
        ikreq.seed_mode = ikreq.SEED_USER
        # if not seed_joints:
        #     seed = JointState()
        #     seed.name = ['right_j0', 'right_j1', 'right_j2', 'right_j3',
        #                  'right_j4', 'right_j5', 'right_j6']
        #     seed.position = [0.7, 0.4, -1.7, 1.4, -1.1, -1.6, -0.4]
        # else:

        # seed = JointState()
        # seed.name = ['right_j0', 'right_j1', 'right_j2', 'right_j3',
        #              'right_j4', 'right_j5', 'right_j6']
        # seed.position = [0.42, -0.38, -1.24, 1.78, 1.16, 1.11, 2.05]

        ###############################
        seed = joint_state_from_cmd(seed_cmd)
        ################################

        ikreq.seed_angles.append(seed)

        # Null space goals are not supported on the baxter
        ## Once the primary IK task is solved, the solver will then try to bias the
        ## the joint angles toward the goal joint configuration. The null space is
        ## the extra degrees of freedom the joints can move without affecting the
        ## primary IK task.
        #ikreq.use_nullspace_goal.append(True)
        ## The nullspace goal can either be the full set or subset of joint angles
        #goal = JointState()
        #goal.name = ['right_j0', 'right_j1', 'right_j2', 'right_j3']
        #goal.position = [0.409, -0.43, -1.2, 1.79]

        #ikreq.nullspace_goal.append(goal)
        ## # The gain used to bias toward the nullspace goal. Must be [0.0, 1.0]
        ## # If empty, the default gain of 0.4 will be used
        ## ikreq.nullspace_gain.append(0.5)
        ## else:
        ## rospy.loginfo("Running Simple IK Service Client example.")
    done, i = False, 0
    while not done and i < 100:
        try:
            rospy.wait_for_service(name_of_service, 5.0)
            resp = iksvc(ikreq)
            done = True
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr("IK service call failed: %s" % (e,))
            i += 1

    if not done:
        raise IOError("IK SERVICE CALL FAILED")

    # Check if result valid, and type of seed ultimately used to get solution
    if (resp.result_type[0] > 0):
        seed_str = {
                    ikreq.SEED_USER: 'User Provided Seed',
                    ikreq.SEED_CURRENT: 'Current Joint Angles',
                    ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                   }.get(resp.result_type[0], 'None')
        # rospy.loginfo("SUCCESS - Valid Joint Solution Found from Seed Type: %s" %
        #       (seed_str,))
        # Format solution into Limb API-compatible dictionary
        limb_joints = dict(list(zip(resp.joints[0].name, resp.joints[0].position)))
        # rospy.loginfo("\nIK Joint Solution:\n%s", limb_joints)
        # rospy.loginfo("------------------")
        # rospy.loginfo("Response Message:\n%s", resp)
        return limb_joints
    else:
        rospy.loginfo("INVALID POSE - No Valid Joint Solution Found.")
        raise ValueError

def get_point_stamped(x,y,z):
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    point = PointStamped(
        header=hdr,
        point=Point(
            x=x,
            y=y,
            z=z,
        )
    )
    return point

def get_pose_stamped(x,y,z,o):
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    p = PoseStamped(
        header=hdr,
        pose=Pose(
            position=Point(
                x=x,
                y=y,
                z=z,
            ),
            orientation=o
        )
    )
    return p

def joint_state_from_cmd(cmd):
    js = JointState()
    js.name = list(cmd.keys())
    js.position = list(cmd.values())
    return js

FORWARD_POINT = Quaternion(
    x=0,
    y=0.707,
    z=0,
    w=0.707,
)

DOWNWARD_POINT = Quaternion(
    x=1,
    y=0,
    z=0,
    w=0,
)

LEFTWARD_POINT = Quaternion(
    x=0.707,
    y=0,
    z=0,
    w=-0.707,
)

LEFTWARD_DIAG = Quaternion(
    x=0.5,
    y=-0.5,
    z=0,
    w=-0.707,
)

EXAMPLE_O = Quaternion(
    x=0.704020578925,
    y=0.710172716916,
    z=0.00244101361829,
    w=0.00194372088834,
)

def main():
    rospy.init_node("inverse_kinematics_test")
    pose = get_pose_stamped(0.45, 0.16, 0.21, EXAMPLE_O)
    print(get_joint_angles(pose))

    pose = get_pose_stamped(0.45, 0.16, 0.21, EXAMPLE_O)
    print(get_joint_angles(pose, limb="left"))

if __name__ == '__main__':
    main()
