from visual_mpc.envs.robot_envs import GripperInterface
import intera_interface
import time


class SawyerDefaultGripper(GripperInterface):
    def __init__(self):
        self._gripper_control = intera_interface.Gripper("right")
        self._gripper_control.calibrate()
        self._gripper_control.set_velocity(self._gripper_control.MIN_VELOCITY)
        self._gripper_control.open()
    
    def get_gripper_state(self, integrate_force=False):
        # returns gripper joint angle, force reading (none if no force)
        return self._gripper_control.get_position(), self._gripper_control.get_force()

    def set_gripper(self, position, wait=False):
        self._gripper_control.set_position(position)
        # just busy wait since the gripper is pretty fast
        while wait and self._gripper_control.is_moving():
            time.sleep(0.1)

    @property
    def GRIPPER_CLOSE(self):
        return self._gripper_control.MIN_POSITION

    @property
    def GRIPPER_OPEN(self):
        return self._gripper_control.MAX_POSITION

    def set_gripper_speed(self, new_speed):
        assert self._gripper_control.MIN_VELOCITY <= new_speed <= self._gripper_control.MAX_VELOCITY
        self._gripper_control.set_velocity(new_speed)
