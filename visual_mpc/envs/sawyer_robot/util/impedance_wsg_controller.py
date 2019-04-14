import rospy
import visual_mpc.foresight_rospkg as foresight_rospkg
from visual_mpc.foresight_rospkg.src.utils.robot_controller import RobotController
from wsg_50_common.msg import Cmd, Status
from threading import Semaphore, Lock
import numpy as np
from visual_mpc.envs.util.interpolation import CSpline
from intera_core_msgs.msg import JointCommand
import cPickle as pkl
import intera_interface
import os


# constants for robot control
NEUTRAL_JOINT_ANGLES = np.array([0.412271, -0.434908, -1.198768, 1.795462, 1.160788, 1.107675, -1.11748145])
NEUTRAL_JOINT_CMD = {k:a for k, a in zip(['right_j{}'.format(i) for i in range(7)], NEUTRAL_JOINT_ANGLES)}
MAX_TIMEOUT = 30
DURATION_PER_POINT = 0.01
N_JOINTS = 7
max_vel_mag = np.array([0.88, 0.678, 0.996, 0.996, 1.776, 1.776, 2.316])
max_accel_mag = np.array([3.5, 2.5, 5, 5, 5, 5, 5])
GRIPPER_CLOSE = 6   # chosen so that gripper closes entirely without pushing against itself
GRIPPER_OPEN = 96   # chosen so that gripper opens entirely without pushing against outer rail
RESET_SKIP = 800


class ImpedanceWSGController(RobotController):
    def __init__(self, control_rate, robot_name, print_debug, gripper_attached=True, send_email=False):
        self.max_release = 0
        self._print_debug = print_debug
        RobotController.__init__(self, robot_name, send_email)
        self.sem_list = [Semaphore(value = 0)]
        self._status_mutex = Lock()

        self._desired_gpos = GRIPPER_OPEN
        self.gripper_speed = 300

        self._force_counter = 0
        self._integrate_gripper_force, self._last_integrate = 0., None
        self.num_timeouts = 0

        self._cmd_publisher = rospy.Publisher('/robot/limb/right/joint_command', JointCommand, queue_size=100)

        self._gripper_attached = gripper_attached
        if gripper_attached:
            self.gripper_pub = rospy.Publisher('/wsg_50_driver/goal_position', Cmd, queue_size=10)
            rospy.Subscriber("/wsg_50_driver/status", Status, self._gripper_callback)
            print("waiting for first status")
            self.sem_list[0].acquire()
            print('gripper initialized!')

            self._navigator = intera_interface.Navigator()
            self._navigator.register_callback(self._close_gripper_handler, 'right_button_ok')

        self.control_rate = rospy.Rate(control_rate)

    def _close_gripper_handler(self, value):
        if value:
            if self._gripper_width >= 40:
                self.close_gripper()    #close gripper on button release
            else:
                self.open_gripper()

    def set_gripper_speed(self, new_speed):
        assert new_speed > 0 and new_speed <= 600, "Speed must be in range (0, 600]"
        self.gripper_speed = new_speed

    def get_gripper_status(self, integrate_force=False):
        if not self._gripper_attached:
            return 0, 0

        self._status_mutex.acquire()
        cum_force, cntr = self._integrate_gripper_force, self._force_counter
        width, force = self._gripper_width, self._gripper_force
        self._integrate_gripper_force = 0.
        self._force_counter = 0
        self._status_mutex.release()

        if integrate_force and cntr > 0:
            self._debug_print("integrating with {} readings, cumulative force: {}".format(cntr, cum_force))
            self._last_integrate = cum_force / cntr
            return width, self._last_integrate
        elif integrate_force and self._last_integrate is not None:
            return width, self._last_integrate

        return width, force

    def _debug_print(self, msg):
        if self._print_debug:
            print(msg)

    def get_limits(self):
        return GRIPPER_CLOSE, GRIPPER_OPEN

    def open_gripper(self, wait = False):
        self.set_gripper(GRIPPER_OPEN, wait=wait)

    def close_gripper(self, wait = False):
        self.set_gripper(GRIPPER_CLOSE, wait=wait)

    def _set_gripper(self, command_pos, wait=False):
        self._status_mutex.acquire()
        self._desired_gpos = command_pos
        if wait:
            if self.num_timeouts > MAX_TIMEOUT:
                rospy.signal_shutdown("MORE THAN {} GRIPPER TIMEOUTS".format(MAX_TIMEOUT))

            sem = Semaphore(value=0)  # use of semaphore ensures script will block if gripper dies during execution
            self.sem_list.append(sem)
            self._status_mutex.release()

            start = rospy.get_time()
            self._debug_print("gripper sem acquire, list len-{}".format(len(self.sem_list)))
            sem.acquire()
            self._debug_print("waited on gripper for {} seconds".format(rospy.get_time() - start))
        else:
            self._status_mutex.release()

    def set_gripper(self, command_pos, wait = False):
        if not self._gripper_attached:
            return

        assert command_pos >= GRIPPER_CLOSE and command_pos <= GRIPPER_OPEN, "Command pos must be in range [GRIPPER_CLOSE, GRIPPER_OPEN]"
        self._set_gripper(command_pos, wait = wait)

    def _gripper_callback(self, status):
        # print('callback! list-len {}, max_release {}'.format(len(self.sem_list), self.max_release))
        self._status_mutex.acquire()

        self._gripper_width, self._gripper_force = status.width, status.force
        self._integrate_gripper_force += status.force
        self._force_counter += 1

        cmd = Cmd()
        cmd.pos = self._desired_gpos
        cmd.speed = self.gripper_speed

        self.gripper_pub.publish(cmd)

        if len(self.sem_list) > 0:
            gripper_close = np.isclose(self._gripper_width, self._desired_gpos, atol=1e-1)

            if gripper_close or self._gripper_force > 0 or self.max_release > 15:
                if self.max_release > 15:
                    self.num_timeouts += 1
                for s in self.sem_list:
                    s.release()
                self.sem_list = []

            self.max_release += 1      #timeout for when gripper responsive but can't acheive commanded state
        else:
            self.max_release = 0

        self._status_mutex.release()

    def neutral_with_impedance(self, duration=2):
        waypoints = [NEUTRAL_JOINT_ANGLES]
        self.move_with_impedance(waypoints, duration)

    def _try_enable(self):
        """
        The start impedance script will try to re-enable the robot once it disables
        The script will wait until that occurs and throw an assertion if it doesn't
        """
        i = 0
        while not self._rs.state().enabled and i < 50:
            rospy.sleep(10)
            i += 1
        assert self._rs.state().enabled, "Robot was disabled, please manually re-enable!"

    def send_pos_command(self, pos):
        self._try_enable()

        command = JointCommand()
        command.mode = JointCommand.POSITION_MODE
        command.names = self.limb.joint_names()
        command.position = pos
        self._cmd_publisher.publish(command)

    def move_with_impedance(self, waypoints, duration=1.5):
        """
        Moves from curent position to final position while hitting waypoints
        :param waypoints: List of arrays containing waypoint joint angles
        :param duration: trajectory duration
        """
        self._try_enable()

        jointnames = self.limb.joint_names()
        prev_joint = np.array([self.limb.joint_angle(j) for j in jointnames])
        waypoints = np.array([prev_joint] + waypoints)

        spline = CSpline(waypoints, duration)

        start_time = rospy.get_time()  # in seconds
        finish_time = start_time + duration  # in seconds

        time = rospy.get_time()
        while time < finish_time:
            pos, velocity, acceleration = spline.get(time - start_time)
            command = JointCommand()
            command.mode = JointCommand.POSITION_MODE
            command.names = jointnames
            command.position = pos
            command.velocity = np.clip(velocity, -max_vel_mag, max_vel_mag)
            command.acceleration = np.clip(acceleration, -max_accel_mag, max_accel_mag)
            self._cmd_publisher.publish(command)

            self.control_rate.sleep()
            time = rospy.get_time()

        for i in xrange(10):
            command = JointCommand()
            command.mode = JointCommand.POSITION_MODE
            command.names = jointnames
            command.position = waypoints[-1]
            self._cmd_publisher.publish(command)

            self.control_rate.sleep()

    def redistribute_objects(self):
        self._debug_print('redistribute...')

        file = '/'.join(str.split(foresight_rospkg.__file__, "/")[
                        :-1]) + '/src/utils/pushback_traj_{}.pkl'.format(self.robot_name)

        self.joint_pos = pkl.load(open(file, "rb"))

        for t in range(0, len(self.joint_pos), RESET_SKIP):
            # print(self.joint_pos[t])
            # self.set_joints(self.joint_pos[t])
            pos_arr = np.array([self.joint_pos[t][j] for j in self.limb.joint_names()])
            self.move_with_impedance([pos_arr])

    def clean_shutdown(self):
        self._send_email("Collection on {} has exited!".format(self.robot_name))
        
        pid = os.getpid()
        print('Exiting example w/ pid: {}'.format(pid))
        os.kill(-pid, 9)
