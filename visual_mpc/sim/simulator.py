import cv2
import shutil
import pickle as pkl
import os
import os.path
import sys
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))


class Sim(object):
    """ Main class to run algorithms and experiments. """

    def __init__(self, config, gpu_id=0, ngpu=1, logger=None, task_mode='train'):
        self._hyperparams = config
        self.agent = config['agent']['type'](config['agent'])
        self.agentparams = config['agent']
        self.policyparams = config['policy']

        self.agentparams['gpu_id'] = gpu_id

        self.policy = config['policy']['type'](self.agent._hyperparams, config['policy'], gpu_id, ngpu)

        self._record_queue = config.pop('record_saver', None)
        self._counter = config.pop('counter', None)

        self.trajectory_list = []
        self.im_score_list = []
        try:
            os.remove(self._hyperparams['agent']['image_dir'])
        except:
            pass
        self.task_mode = task_mode

    def run(self):
        if self._counter is None:
            for i in range(self._hyperparams['start_index'], self._hyperparams['end_index']+1):
                self.take_sample(i)
        else:
            itr = self._counter.ret_increment()
            while itr < self._hyperparams['ntraj']:
                print('taking sample {} of {}'.format(itr, self._hyperparams['ntraj']))
                self.take_sample(itr)
                itr = self._counter.ret_increment()
        self.agent.cleanup()

    def take_sample(self, sample_index):
        self.policy.reset()
        agent_data, obs_dict, policy_out = self.agent.sample(self.policy, sample_index)
        if self._hyperparams.get('save_data', True):
            self.save_data(sample_index, agent_data, obs_dict, policy_out)
        return agent_data

    def save_data(self, itr, agent_data, obs_dict, policy_outputs):
        if self._hyperparams.get('save_only_good', False) and not agent_data['goal_reached']:
            return

        if self._hyperparams.get('save_raw_images', False):
            self._save_raw_data(itr, agent_data, obs_dict, policy_outputs)
        elif self._record_queue is not None:
            self._record_queue.put((agent_data, obs_dict, policy_outputs))
        else:
            raise ValueError('Saving neither raw data nor records')

    def _save_raw_data(self, itr, agent_data, obs_dict, policy_outputs):
        data_save_dir = self.agentparams['data_save_dir']

        ngroup = self._hyperparams.get('ngroup', 1000)
        igrp = itr // ngroup
        group_folder = data_save_dir + '/{}/traj_group{}'.format(self.task_mode, igrp)
        if not os.path.exists(group_folder):
            os.makedirs(group_folder)

        traj_folder = group_folder + '/traj{}'.format(itr)
        if os.path.exists(traj_folder):
            print('trajectory folder {} already exists, deleting the folder'.format(traj_folder))
            shutil.rmtree(traj_folder)

        os.makedirs(traj_folder)
        print('writing: ', traj_folder)
        if 'images' in obs_dict:
            images = obs_dict.pop('images')
            T, n_cams = images.shape[:2]
            for i in range(n_cams):
                os.mkdir(traj_folder + '/images{}'.format(i))
            for t in range(T):
                for i in range(n_cams):
                    cv2.imwrite('{}/images{}/im_{}.png'.format(traj_folder, i, t), images[t, i, :, :, ::-1])
        with open('{}/agent_data.pkl'.format(traj_folder), 'wb') as file:
            pkl.dump(agent_data, file)
        with open('{}/obs_dict.pkl'.format(traj_folder), 'wb') as file:
            pkl.dump(obs_dict, file)
        with open('{}/policy_out.pkl'.format(traj_folder), 'wb') as file:
            pkl.dump(policy_outputs, file)
