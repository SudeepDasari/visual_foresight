from .robot_controller_interface import RobotController
from .grippers.gripper import GripperInterface


def get_controller_class(robot_type):
    if robot_type == 'sawyer':
        from .sawyer.sawyer_impedance import SawyerImpedanceController
        return SawyerImpedanceController
    else:
        raise NotImplementedError
