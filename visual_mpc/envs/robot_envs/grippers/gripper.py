import logging


class GripperInterface(object):
    """
    Interface that all grippers should provide functions for
        - All grippers should have a corresponding class which overrides this interface
        - GripperInterface instances act as if 'no' gripper is attached
    """
    def __init__(self):
        logging.getLogger('robot_logger').info('Creating gripper object')

    def get_gripper_state(self, integrate_force=False):
        # returns gripper joint angle, force reading (none if no force)
        logging.getLogger('robot_logger').debug("Attempting to get non-existent gripper's state!")
        return 0.0, None

    def get_gripper_limits(self):
        return self.GRIPPER_CLOSE, self.GRIPPER_OPEN

    def set_gripper(self, position, wait=False):
        logging.getLogger('robot_logger').debug('Calling set_gripper on non-existent gripper!')

    def open_gripper(self, wait = False):
        self.set_gripper(self.GRIPPER_OPEN, wait=wait)

    def close_gripper(self, wait = False):
        self.set_gripper(self.GRIPPER_CLOSE, wait=wait)

    @property
    def GRIPPER_CLOSE(self):
        return 0

    @property
    def GRIPPER_OPEN(self):
        return 0

    def set_gripper_speed(self, new_speed):
        logging.getLogger('robot_logger').debug('Calling set_gripper_speed on non-existent gripper!')
