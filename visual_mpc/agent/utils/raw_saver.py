import os
import shutil
import pickle as pkl
import cv2


class RawSaver():
    def __init__(self, save_dir, ngroup=1000):
        self.save_dir = save_dir
        self.ngroup = ngroup

    def save_traj(self, itr, agent_data=None, obs_dict=None, policy_outputs=None):

        igrp = itr // self.ngroup
        group_folder = self.save_dir + '/raw/traj_group{}'.format(igrp)
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

        if agent_data is not None:
            with open('{}/agent_data.pkl'.format(traj_folder), 'wb') as file:
                pkl.dump(agent_data, file)
        if obs_dict is not None:
            with open('{}/obs_dict.pkl'.format(traj_folder), 'wb') as file:
                pkl.dump(obs_dict, file)
        if policy_outputs is not None:
            with open('{}/policy_out.pkl'.format(traj_folder), 'wb') as file:
                pkl.dump(policy_outputs, file)