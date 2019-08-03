""" This file defines an agent for the MuJoCo simulator environment. """
import pdb
import copy
import numpy as np
from visual_mpc.policy import get_policy_args
from visual_mpc.utils.im_utils import resize_store
from .utils.file_saver import start_file_worker


class Bad_Traj_Exception(Exception):
    def __init__(self):
        pass


class Image_Exception(Exception):
    def __init__(self):
        pass


class Environment_Exception(Exception):
    def __init__(self):
        pass


class GeneralAgent(object):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """

    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        self.T = self._hyperparams['T']
        self._goal_obj_pose = None
        self._goal_image = None
        self._reset_state = None
        self._is_robot = 'robot_name' in hyperparams['env'][1]
        self._save_worker = start_file_worker()
        self._setup_world(0)

    def _setup_world(self, itr):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        """
        env_type, env_params = self._hyperparams['env']
        self.env = env_type(env_params, self._reset_state)

        self._hyperparams['adim'] = self.adim = self.env.adim
        self._hyperparams['sdim'] = self.sdim = self.env.sdim
        self._hyperparams['ncam'] = self.ncam = self.env.ncam
        self.num_objects = self.env.num_objects

    def sample(self, policy, i_traj):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        """
        if not self._is_robot:
            if "gen_xml" in self._hyperparams:
                if i_traj % self._hyperparams['gen_xml'] == 0 and i_traj > 0:
                    self._setup_world(i_traj)
            elif i_traj > 0:
                self._setup_world(i_traj)

        traj_ok, obs_dict, policy_outs, agent_data = False, None, None, None
        i_trial = 0

        while not traj_ok and i_trial < self._hyperparams.get('imax', 100):
            i_trial += 1
            try:
                agent_data, obs_dict, policy_outs = self.rollout(policy, i_trial, i_traj)
                traj_ok = agent_data['traj_ok']
            except Image_Exception, Environment_Exception:
                traj_ok = False

        if not traj_ok:
            raise Bad_Traj_Exception

        print('needed {} trials'.format(i_trial))

        return agent_data, obs_dict, policy_outs

    def _post_process_obs(self, env_obs, agent_data, initial_obs=False):
        """
        Handles conversion from the environment observations, to agent observation
        space. Observations are accumulated over time, and images are resized to match
        the given image_heightximage_width dimensions.

        Original images from cam index 0 are added to buffer for saving gifs (if needed)

        Data accumlated over time is cached into an observation dict and returned. Data specific to each
        time-step is returned in agent_data

        :param env_obs: observations dictionary returned from the environment
        :param initial_obs: Whether or not this is the first observation in rollout
        :return: obs: dictionary of observations up until (and including) current timestep
        """
        agent_img_height = self._hyperparams['image_height']
        agent_img_width = self._hyperparams['image_width']

        if initial_obs:
            T = self._hyperparams['T'] + 1
            self._agent_cache = {}
            for k in env_obs:
                if k == 'images':
                    if 'obj_image_locations' in env_obs:
                        self.traj_points = []
                    n_cams = env_obs['images'].shape[0]
                    self._agent_cache['images'] = np.zeros((T, n_cams, agent_img_height, agent_img_width, 3),
                                                           dtype=np.uint8)
                elif isinstance(env_obs[k], np.ndarray):
                    obs_shape = [T] + list(env_obs[k].shape)
                    self._agent_cache[k] = np.zeros(tuple(obs_shape), dtype=env_obs[k].dtype)
                else:
                    self._agent_cache[k] = []
            self._cache_cntr = 0

        t = self._cache_cntr
        self._cache_cntr += 1

        point_target_width = float(self._hyperparams.get('point_space_width', agent_img_width))
        obs = {}
        for k in env_obs:
            if k == 'images':
                self.large_images_traj.append(env_obs['images'][0])  # only take first camera
                resize_store(t, self._agent_cache['images'], env_obs['images'])

            elif k == 'obj_image_locations':
                self.traj_points.append(copy.deepcopy(env_obs['obj_image_locations'][0]))  # only take first camera
                env_obs['obj_image_locations'] = np.round((env_obs['obj_image_locations'] *
                                                           point_target_width / env_obs['images'].shape[2])).astype(
                    np.int64)
                self._agent_cache['obj_image_locations'][t] = env_obs['obj_image_locations']
            elif isinstance(env_obs[k], np.ndarray):
                self._agent_cache[k][t] = env_obs[k]
            else:
                self._agent_cache[k].append(env_obs[k])
            obs[k] = self._agent_cache[k][:self._cache_cntr]

        if 'obj_image_locations' in env_obs:
            agent_data['desig_pix'] = env_obs['obj_image_locations']
        if self._goal_image is not None:
            agent_data['goal_image'] = self._goal_image
        if self._goal_obj_pose is not None:
            agent_data['goal_pos'] = self._goal_obj_pose
            agent_data['goal_pix'] = self.env.get_goal_pix(point_target_width)
        if self._reset_state is not None:
            agent_data['reset_state'] = self._reset_state
            obs['reset_state'] = self._reset_state

        return obs

    def _required_rollout_metadata(self, agent_data, traj_ok, t, i_traj, i_tr, reset_state):
        """
        Adds meta_data into the agent dictionary that is MANDATORY for later parts of pipeline
        :param agent_data: Agent data dictionary
        :param traj_ok: Whether or not rollout succeeded
        :return: None
        """
        agent_data['term_t'] = t - 1
        if self.env.has_goal():
            agent_data['goal_reached'] = self.env.goal_reached()
        agent_data['traj_ok'] = traj_ok

        if self._hyperparams.get('save_reset_data', False):
            agent_data['reset_state'] = reset_state

        if 'make_final_recording' in self._hyperparams:
            self._save_worker.put(('path', self.record_path))
            self.env.save_recording(self._save_worker, i_traj)

    def rollout(self, policy, i_trial, i_traj):
        """
        Rolls out policy for T timesteps
        :param policy: Class extending abstract policy class. Must have act method (see arg passing details)
        :param i_trial: Rollout attempt index (increment each time trajectory fails rollout)
        :return: - agent_data: Dictionary of extra statistics/data collected by agent during rollout
                 - obs: dictionary of environment's observations. Each key maps to that values time-history
                 - policy_ouputs: list of policy's outputs at each timestep.
                 Note: tfrecord saving assumes all keys in agent_data/obs/policy_outputs point to np arrays or primitive int/float
        """
        self._init()

        agent_data, policy_outputs = {}, []

        # Take the sample.
        t = 0
        done = self._hyperparams['T'] <= 0
        initial_env_obs, reset_state = self.env.reset()
        obs = self._post_process_obs(initial_env_obs, agent_data, True)
        policy.reset()

        while not done:
            """
            Every time step send observations to policy, acts in environment, and records observations

            Policy arguments are created by
                - populating a kwarg dict using get_policy_arg
                - calling policy.act with given dictionary

            Policy returns an object (pi_t) where pi_t['actions'] is an action that can be fed to environment
            Environment steps given action and returns an observation
            """
            pi_t = policy.act(**get_policy_args(policy, obs, t, i_traj, agent_data))
            policy_outputs.append(pi_t)

            obs = self._post_process_obs(self.env.step(copy.deepcopy(pi_t['actions'])), agent_data)

            if 'rejection_sample' in self._hyperparams and 'rejection_end_early' in self._hyperparams:
                print('traj rejected!')
                if self._hyperparams['rejection_sample'] > i_trial and not self.env.goal_reached():
                    return {'traj_ok': False}, None, None

            if (self._hyperparams['T'] - 1) == t:
                done = True
            t += 1

        traj_ok = self.env.valid_rollout()
        if 'rejection_sample' in self._hyperparams:
            if self._hyperparams['rejection_sample'] > i_trial:
                assert self.env.has_goal(), 'Rejection sampling enabled but env has no goal'
                traj_ok = self.env.goal_reached()
            print('goal_reached', self.env.goal_reached())

        self._required_rollout_metadata(agent_data, traj_ok, t, i_traj, i_trial, reset_state)
        return agent_data, obs, policy_outputs

    def _init(self):
        """
        Set the world to a given model
        """
        self.large_images_traj, self.traj_points = [], None

    def cleanup(self):
        print('Cleaning up file saver....')
        self._save_worker.put(None)
        self._save_worker.join()

    @property
    def record_path(self):
        return self._hyperparams['data_save_dir'] + '/record/'
