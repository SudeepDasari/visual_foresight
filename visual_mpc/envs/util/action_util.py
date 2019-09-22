import numpy as np


def autograsp_grip_logic(gripper_zpos, zthresh, gripper_closed, reopen, grasp_condition):
    if gripper_zpos < zthresh:
        gripper_closed = True
    elif reopen and not grasp_condition:
        gripper_closed = False
    return gripper_closed


def autograsp_dynamics(prev_target_qpos, action, gripper_closed, gripper_zpos, zthresh, reopen, grasp_condition, open_action=-1, close_action=1):
    target_qpos = np.zeros_like(prev_target_qpos)
    target_qpos[:4] = action[:4] + prev_target_qpos[:4]

    gripper_closed = autograsp_grip_logic(gripper_zpos, zthresh, gripper_closed, reopen, grasp_condition)

    if gripper_closed:
        target_qpos[4] = close_action
    else:
        target_qpos[4] = open_action

    return target_qpos, gripper_closed
