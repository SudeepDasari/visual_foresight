from .general_agent import GeneralAgent
from visual_mpc.utils.im_utils import resize_store
import pickle as pkl
import numpy as np
import cv2
import os
import shutil


class BenchmarkAgent(GeneralAgent):
    def __init__(self, hyperparams):
        self._start_goal_confs = hyperparams.get('start_goal_confs', None)
        if 'camera_topics' in hyperparams['env'][1]:
            self.ncam = len(hyperparams['env'][1]['camera_topics'])
        else:
            self.ncam = hyperparams['env'][0].default_ncam()
        GeneralAgent.__init__(self, hyperparams)

        if not self._is_robot:
            self._hyperparams['gen_xml'] = 1

    def _post_process_obs(self, env_obs, agent_data, initial_obs=False):
        obs = super(BenchmarkAgent, self)._post_process_obs(env_obs, agent_data, initial_obs)
        agent_data['verbose_worker'] = self._save_worker
        return obs

    def _setup_world(self, itr):
        old_ncam = self.ncam
        self._reset_state = self._load_raw_data(itr)
        GeneralAgent._setup_world(self, itr)
        assert old_ncam == self.ncam, """Environment has {} cameras but benchmark has {}. 
                                            Feed correct ncam in agent_params""".format(self.ncam, old_ncam)

    def _required_rollout_metadata(self, agent_data, traj_ok, t, i_traj, i_itr, reset_state):
        GeneralAgent._required_rollout_metadata(self, agent_data, traj_ok, t, i_traj, i_itr, reset_state)
        point_target_width = self._hyperparams.get('point_space_width', self._hyperparams['image_width'])
        ntasks = self._hyperparams.get('ntask', 1)
        if 'no_goal_def' not in self._hyperparams:
            agent_data['stats'] = self.env.eval(point_target_width, self._hyperparams.get('_bench_save', None), ntasks)

        if not traj_ok and self._is_robot:
            """
            Hot-wire traj_ok to give user chance to abort experiment on failure
            """
            print('WARNING TRAJ FAILED')
            if 'n' in raw_input('would you like to retry? (y/n): '):    # is fine since robot_bench only runs in py2
                agent_data['traj_ok'] = True

    def _init(self):
        if self._is_robot:
            if '_bench_save' not in self._hyperparams:
                raise ValueError("Benchmark dir missing! Maybe you didn't add --benchmark flag?")

            done = False
            while not done:
                if os.path.exists(self._hyperparams['_bench_save']):
                    shutil.rmtree(self._hyperparams['_bench_save'])
                os.makedirs(self._hyperparams['_bench_save'])

                ntasks = self._hyperparams.get('ntask', 1)

                if 'no_goal_def' not in self._hyperparams:
                    if 'register_gtruth' in self._hyperparams and len(self._hyperparams['register_gtruth']) == 2:
                        raw_goal_image, self._goal_obj_pose = self.env.get_obj_desig_goal(self._hyperparams['_bench_save'], True,
                                                                                           ntasks=ntasks)
                        goal_dims = (1, self.ncam, self._hyperparams['image_height'], self._hyperparams['image_width'], 3)
                        self._goal_image = np.zeros(goal_dims, dtype=np.uint8)
                        resize_store(0, self._goal_image, raw_goal_image)
                        self._goal_image = self._goal_image.astype(np.float32) / 255.
                    else:
                        self._goal_obj_pose = self.env.get_obj_desig_goal(self._hyperparams['_bench_save'], ntasks=ntasks)
                else:
                    if 'goal_image_only' in self._hyperparams:
                        raw_goal_image = self.env.get_goal_image(self._hyperparams['_bench_save'])
                        goal_dims = (1, self.ncam, self._hyperparams['image_height'], self._hyperparams['image_width'], 3)
                        self._goal_image = np.zeros(goal_dims, dtype=np.uint8)
                        resize_store(0, self._goal_image, raw_goal_image)
                        self._goal_image = self._goal_image.astype(np.float32) / 255.
                    elif 'load_goal_image' in self._hyperparams:
                        import scipy
                        im = scipy.misc.imread(self._hyperparams['load_goal_image'])
                        goal_dims = (1, self.ncam, self._hyperparams['image_height'], self._hyperparams['image_width'], 3)
                        self._goal_image = np.zeros(goal_dims, dtype=np.uint8)
                        resize_store(0, self._goal_image, im[None])
                        self._goal_image = self._goal_image.astype(np.float32) / 255.
                    else:
                        raise NotImplementedError

                if 'no_goal_def' in self._hyperparams or 'y' in raw_input('Is definition okay? (y/n):'):
                    done = True
                    self._save_worker.put(('path', self._hyperparams['_bench_save']))

            return GeneralAgent._init(self)

        self.env.set_goal_obj_pose(self._goal_obj_pose)
        return GeneralAgent._init(self)

    def _load_raw_data(self, itr):
        """
        doing the reverse of save_raw_data
        :param itr:
        :return:
        """
        if self._is_robot:   # robot experiments don't have a reset state
            return None

        itr = self._hyperparams.get('iex', itr)

        ngroup = 1000
        igrp = itr // ngroup
        group_folder = '{}/traj_group{}'.format(self._start_goal_confs, igrp)
        traj_folder = group_folder + '/traj{}'.format(itr)

        print('reading from: ', traj_folder)
        num_images = self._hyperparams.get('num_load_steps', 2)

        obs_dict = {}
        goal_images = np.zeros([num_images, self.ncam, self._hyperparams['image_height'], self._hyperparams['image_width'], 3])
        for t in range(num_images):  #TODO detect number of images automatically in folder
            for i in range(self.ncam):
                image_file = '{}/images{}/im_{}.png'.format(traj_folder, i, t)
                if not os.path.isfile(image_file):
                    raise ValueError("Can't find goal image: {}".format(image_file))
                goal_images[t, i] = cv2.imread(image_file)[...,::-1]

        self._goal_image = goal_images.astype(np.float32)/255.

        with open('{}/agent_data.pkl'.format(traj_folder), 'rb') as file:
            agent_data = pkl.load(file)
        with open('{}/obs_dict.pkl'.format(traj_folder), 'rb') as file:
            obs_dict.update(pkl.load(file))
        reset_state = agent_data['reset_state']

        self._goal_obj_pose = obs_dict['object_qpos'][-1]

        verbose_dir = '{}/verbose/traj_{}'.format(self._hyperparams['data_save_dir'], itr)
        self._save_worker.put(('path', verbose_dir))

        return reset_state

    @property
    def record_path(self):
        if self._is_robot:
            return self._hyperparams['_bench_save']
        else:
            return self._hyperparams['data_save_dir'] + '/record/'
