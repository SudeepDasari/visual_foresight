'''
The default kuka gripper is a rigid rod
'''

from visual_mpc.envs.robot_envs import GripperInterface
import time
from visual_mpc.envs.robot_envs.kuka.kuka_interface import KukaInterface

class KukaDefaultGripper(GripperInterface):
    def __init__(self, arm="kuka_arm"):

        self.kuka_obj = KukaInterface()

    def get_gripper_state(self, integrate_force=False):
        # returns gripper joint angle, force reading (none if no force)
        ## ** get gripper joint angle and force on the gripper ***###
        return self.kuka_obj.get_kuka_joint_angles()[6],self.kuka_obj.get_kuka_joint_angles_effort()[6]

    def set_gripper(self, position, wait=False):
        pass
        # self._gripper_control.command_position(position)
        # # just busy wait since the gripper is pretty fast
        # while wait and self._gripper_control.moving():
        #     time.sleep(0.1)

    @property
    def GRIPPER_CLOSE(self):
        return 0.0

    @property
    def GRIPPER_OPEN(self):
        return 0.0

    def set_gripper_speed(self, new_speed):
        pass
        # assert 0.0 <= new_speed <= 100.0
        # self._gripper_control.set_velocity(new_speed)
