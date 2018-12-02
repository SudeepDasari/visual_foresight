import rospy
from robot_controller import RobotController
from wsg_50_common.msg import Cmd, Status
from threading import Semaphore, Lock

GRIPPER_CLOSE = 6   #chosen so that gripper closes entirely without pushing against itself
GRIPPER_OPEN = 96   #chosen so that gripper opens entirely without pushing against outer rail

from std_msgs.msg import Float32, Int64
from sensor_msgs.msg import JointState
import numpy as np

import python_visual_mpc.visual_mpc_core.envs.sawyer_robot.visual_mpc_rospkg as visual_mpc_rospkg

import cPickle as pickle
import intera_interface

NEUTRAL_JOINT_ANGLES =[0.412271, -0.434908, -1.198768, 1.795462, 1.160788, 1.107675, 2.068076]
MAX_TIMEOUT = 30


class WSGRobotController(RobotController):
    def __init__(self, control_rate, robot_name):
        self.max_release = 0
        RobotController.__init__(self)
        self.sem_list = [Semaphore(value = 0)]
        self._status_mutex = Lock()
        self.robot_name = robot_name

        self._desired_gpos = GRIPPER_OPEN
        self.gripper_speed = 300

        self._force_counter = 0
        self._integrate_gripper_force = 0.
        self.num_timeouts = 0

        self.gripper_pub = rospy.Publisher('/wsg_50_driver/goal_position', Cmd, queue_size=10)
        rospy.Subscriber("/wsg_50_driver/status", Status, self._gripper_callback)

        print("waiting for first status")
        self.sem_list[0].acquire()
        print('gripper initialized!')

        self.imp_ctrl_publisher = rospy.Publisher('/desired_joint_pos', JointState, queue_size=1)
        self.imp_ctrl_release_spring_pub = rospy.Publisher('/release_spring', Float32, queue_size=10)
        self.imp_ctrl_active = rospy.Publisher('/imp_ctrl_active', Int64, queue_size=10)

        self.control_rate = rospy.Rate(control_rate)

        self.imp_ctrl_release_spring(100)
        self.imp_ctrl_active.publish(1)

        self._navigator = intera_interface.Navigator()
        self._navigator.register_callback(self._close_gripper_handler, 'right_button_ok')

    def _close_gripper_handler(self, value):
        if value:
            self.close_gripper()    #close gripper on button release

    def set_gripper_speed(self, new_speed):
        assert new_speed > 0 and new_speed <= 600, "Speed must be in range (0, 600]"
        self.gripper_speed = new_speed

    def get_gripper_status(self, integrate_force=False):
        self._status_mutex.acquire()
        cum_force, cntr = self._integrate_gripper_force, self._force_counter
        width, force = self.gripper_width, self.gripper_force
        self._integrate_gripper_force = 0.
        self._force_counter = 0
        self._status_mutex.release()

        if integrate_force and cntr > 0:
            print("integrating with {} readings, cumulative force: {}".format(cntr, cum_force))
            return width, cum_force / cntr

        return width, force

    def get_limits(self):
        return GRIPPER_CLOSE, GRIPPER_OPEN

    def open_gripper(self, wait = False):
        self.set_gripper(GRIPPER_OPEN, wait = wait)

    def close_gripper(self, wait = False):
        self.set_gripper(GRIPPER_CLOSE, wait = wait)

    def _set_gripper(self, command_pos, wait = False):
        self._desired_gpos = command_pos
        if wait:
            if self.num_timeouts > MAX_TIMEOUT:
                rospy.signal_shutdown("MORE THAN {} GRIPPER TIMEOUTS".format(MAX_TIMEOUT))

            sem = Semaphore(value=0)  # use of semaphore ensures script will block if gripper dies during execution

            self._status_mutex.acquire()
            self.sem_list.append(sem)
            self._status_mutex.release()

            start = rospy.get_time()
            print("gripper sem acquire, list len-{}".format(len(self.sem_list)))
            sem.acquire()
            print("waited on gripper for {} seconds".format(rospy.get_time() - start))

    def set_gripper(self, command_pos, wait = False):
        assert command_pos >= GRIPPER_CLOSE and command_pos <= GRIPPER_OPEN, "Command pos must be in range [GRIPPER_CLOSE, GRIPPER_OPEN]"
        self._set_gripper(command_pos, wait = wait)

    def _gripper_callback(self, status):
        # print('callback! list-len {}, max_release {}'.format(len(self.sem_list), self.max_release))
        self._status_mutex.acquire()

        self.gripper_width, self.gripper_force = status.width, status.force
        self._integrate_gripper_force += status.force
        self._force_counter += 1

        cmd = Cmd()
        cmd.pos = self._desired_gpos
        cmd.speed = self.gripper_speed

        self.gripper_pub.publish(cmd)

        if len(self.sem_list) > 0:
            gripper_close = np.isclose(self.gripper_width, self._desired_gpos, atol=1e-1)

            if gripper_close or self.gripper_force > 0 or self.max_release > 15:
                if self.max_release > 15:
                    self.num_timeouts += 1
                for s in self.sem_list:
                    s.release()
                self.sem_list = []

            self.max_release += 1      #timeout for when gripper responsive but can't acheive commanded state
        else:
            self.max_release = 0

        self._status_mutex.release()

    def reset_with_impedance(self, angles = NEUTRAL_JOINT_ANGLES, duration= 3., open_gripper = True, close_first = False, stiffness = 150, reset_sitffness = 100):
        if open_gripper:
            if close_first:
                self._set_gripper(2, wait=True)
            self._set_gripper(100, wait=True)
            self.open_gripper()

        self.imp_ctrl_release_spring(reset_sitffness)
        self.move_to_joints_impedance_sec(angles, duration=duration)
        self.imp_ctrl_release_spring(stiffness)

        self.get_gripper_status()  # dummy call to flush integration of gripper force

    def disable_impedance(self):
        self.imp_ctrl_active.publish(0)

    def enable_impedance(self):
        self.imp_ctrl_active.publish(1)

    def imp_ctrl_release_spring(self, maxstiff):
        self.imp_ctrl_release_spring_pub.publish(maxstiff)

    def move_to_joints_impedance_sec(self, joint_angle_array, duration = 2.):
        cmd = dict(list(zip(self.limb.joint_names(), joint_angle_array)))
        self.move_with_impedance_sec(cmd, duration)

    def move_with_impedance(self, des_joint_angles):
        """
        non-blocking
        """
        js = JointState()
        js.name = self.limb.joint_names()
        js.position = [des_joint_angles[n] for n in js.name]
        self.imp_ctrl_publisher.publish(js)

    def move_with_impedance_sec(self, cmd, duration=2.):
        jointnames = self.limb.joint_names()
        prev_joint = [self.limb.joint_angle(j) for j in jointnames]
        new_joint = np.array([cmd[j] for j in jointnames])

        start_time = rospy.get_time()  # in seconds
        finish_time = start_time + duration  # in seconds


        while rospy.get_time() < finish_time:
            int_joints = prev_joint + (rospy.get_time()-start_time)/(finish_time-start_time)*(new_joint-prev_joint)
            # print int_joints
            cmd = dict(list(zip(self.limb.joint_names(), list(int_joints))))
            self.move_with_impedance(cmd)
            self.control_rate.sleep()

    def redistribute_objects(self):
        self.reset_with_impedance(duration=1.5)
        print('redistribute...')

        file = '/'.join(str.split(visual_mpc_rospkg.__file__, "/")[
                        :-1]) + '/src/utils/pushback_traj_{}.pkl'.format(self.robot_name)

        self.joint_pos = pickle.load(open(file, "rb"))

        self.imp_ctrl_release_spring(100)
        self.imp_ctrl_active.publish(1)

        replay_rate = rospy.Rate(700)
        for t in range(len(self.joint_pos)):
            print('step {0} joints: {1}'.format(t, self.joint_pos[t]))
            replay_rate.sleep()
            self.move_with_impedance(self.joint_pos[t])