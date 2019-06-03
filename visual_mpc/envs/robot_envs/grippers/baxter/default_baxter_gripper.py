from visual_mpc.envs.robot_envs import GripperInterface
import baxter_interface
import time
from baxter_interface import CHECK_VERSION

class BaxterDefaultGripper(GripperInterface):
    def __init__(self, arm="right"):
        self._gripper_control = baxter_interface.Gripper("right", CHECK_VERSION)
        self._gripper_control.calibrate()
        self._gripper_control.set_velocity(50)
        self._gripper_control.open()
    
    def get_gripper_state(self, integrate_force=False):
        # returns gripper joint angle, force reading (none if no force)
        return self._gripper_control.position(), self._gripper_control.force()

    def set_gripper(self, position, wait=False):
        self._gripper_control.command_position(position)
        # just busy wait since the gripper is pretty fast
        while wait and self._gripper_control.moving():
            time.sleep(0.1)

    @property
    def GRIPPER_CLOSE(self):
        return 0.0

    @property
    def GRIPPER_OPEN(self):
        return 100.0

    def set_gripper_speed(self, new_speed):
        assert 0.0 <= new_speed <= 100.0
        self._gripper_control.set_velocity(new_speed)
