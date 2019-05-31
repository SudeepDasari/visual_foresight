import logging


class GripperInterface(object):
    def __init__(self):
        logging.getLogger('robot_logger').info('Creating gripper object')

    def get_gripper_state(self, integrate_force=False):
        # returns gripper joint angle, force reading (none if no force)
        raise NotImplementedError

    def get_gripper_limits(self):
        return self.GRIPPER_CLOSE, self.GRIPPER_OPEN

    def set_gripper(self, position, wait=False):
        raise NotImplementedError

    def open_gripper(self, wait = False):
        raise NotImplementedError

    def close_gripper(self, wait = False):
        raise NotImplementedError

    @property
    def GRIPPER_CLOSE(self):
        raise NotImplementedError

    @property
    def GRIPPER_OPEN(self):
        raise NotImplementedError

    def set_gripper_speed(self, new_speed):
        raise NotImplementedError
