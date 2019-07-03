from .robot_controller_interface import RobotController
from .grippers.gripper import GripperInterface


def get_controller_class(robot_type):
    if robot_type == 'sawyer':
        from .sawyer.sawyer_impedance import SawyerImpedanceController
        return SawyerImpedanceController
    elif robot_type == 'widowx':
        from .widowx.widowx_controller import WidowXController
        return WidowXController
    elif robot_type == 'baxter':
        from .baxter.baxter_impedance import BaxterImpedanceController
        return BaxterImpedanceController
    else:
        raise NotImplementedError
