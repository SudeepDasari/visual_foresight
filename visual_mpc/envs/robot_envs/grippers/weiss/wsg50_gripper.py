import rospy
from wsg_50_common.msg import Cmd, Status
from visual_mpc.envs.robot_envs import GripperInterface
import logging
from threading import Semaphore, Lock, Thread
import time
import numpy as np


GRIPPER_CLOSE = 6   # chosen so that gripper closes entirely without pushing against itself
GRIPPER_OPEN = 96   # chosen so that gripper opens entirely without pushing against outer rail
ROS_NODE_TIMEOUT = 600     # kill script if waiting for more than 10 minutes on gripper
MAX_TIMEOUT = 10


class WSG50Gripper(GripperInterface):
    def __init__(self):
        super(WSG50Gripper, self).__init__()
        self.max_release = 0
        self.sem_list = [Semaphore(value = 0)]
        self._status_mutex = Lock()

        self._desired_gpos = GRIPPER_OPEN
        self._gripper_speed = 300

        self._force_counter = 0
        self._integrate_gripper_force, self._last_integrate = 0., None
        self._last_status_t = time.time()
        self.num_timeouts = 0

        self.gripper_pub = rospy.Publisher('/wsg_50_driver/goal_position', Cmd, queue_size=10)
        rospy.Subscriber("/wsg_50_driver/status", Status, self._gripper_callback)
        logging.getLogger('robot_logger').info("waiting for first status")
        self.sem_list[0].acquire()
        logging.getLogger('robot_logger').info('gripper initialized!')

        self._bg = Thread(target=self._background_monitor)
        self._bg.start()
    
    def _background_monitor(self):
        while True:
            self._status_mutex.acquire()
            if len(self.sem_list) > 0 and time.time() - self._last_status_t >= ROS_NODE_TIMEOUT:
                logging.getLogger('robot_logger').error('No gripper messages in {} seconds, maybe the node crashed?'.format(ROS_NODE_TIMEOUT))
                self.clean_shutdown()
            self._status_mutex.release()
            time.sleep(30)


    def get_gripper_state(self, integrate_force=False):
        self._status_mutex.acquire()
        cum_force, cntr = self._integrate_gripper_force, self._force_counter
        width, force = self._gripper_width, self._gripper_force
        self._integrate_gripper_force = 0.
        self._force_counter = 0
        self._status_mutex.release()

        if integrate_force and cntr > 0:
            logging.getLogger('robot_logger').debug("integrating with {} readings, cumulative force: {}".format(cntr, cum_force))
            self._last_integrate = cum_force / cntr
            return width, self._last_integrate
        elif integrate_force and self._last_integrate is not None:
            return width, self._last_integrate

        return width, force

    def get_gripper_limits(self):
        return self.GRIPPER_CLOSE, self.GRIPPER_OPEN

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
            logging.getLogger('robot_logger').debug("gripper sem acquire, list len-{}".format(len(self.sem_list)))
            sem.acquire()
            logging.getLogger('robot_logger').debug("waited on gripper for {} seconds".format(rospy.get_time() - start))
        else:
            self._status_mutex.release()

    def set_gripper(self, command_pos, wait = False):
        assert command_pos >= GRIPPER_CLOSE and command_pos <= GRIPPER_OPEN, "Command pos must be in range [GRIPPER_CLOSE, GRIPPER_OPEN]"
        self._set_gripper(command_pos, wait = wait)

    @property
    def GRIPPER_CLOSE(self):
        return GRIPPER_CLOSE

    @property
    def GRIPPER_OPEN(self):
        return GRIPPER_OPEN

    def set_gripper_speed(self, new_speed):
        assert new_speed > 0 and new_speed <= 600, "Speed must be in range (0, 600]"
        self._gripper_speed = new_speed

    def _gripper_callback(self, status):
        # print('callback! list-len {}, max_release {}'.format(len(self.sem_list), self.max_release))
        self._status_mutex.acquire()
        self._gripper_width, self._gripper_force = status.width, status.force
        self._integrate_gripper_force += status.force
        self._force_counter += 1

        cmd = Cmd()
        cmd.pos = self._desired_gpos
        cmd.speed = self._gripper_speed

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

        self._last_status_t = time.time()
        self._status_mutex.release()
