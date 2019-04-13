from visual_mpc.datasets.base_dataset import BaseVideoDataset
import h5py
import tensorflow as tf
import numpy as np
import cv2
import glob
import random


class HDF5VideoDataset(BaseVideoDataset):
    def _init_dataset(self):
        # read hdf5 from base dir and check contents
        dataset_contents = glob.glob(self._base_dir + '/*.hdf5')
        assert len(dataset_contents), "No hdf5 files in dataset!"

        # consistent shuffling of dataset contents according to set RNG
        dataset_contents.sort()
        random.Random(self._hparams.RNG).shuffle(dataset_contents)

        rand_file = dataset_contents[0]
        with h5py.File(rand_file, 'r') as f:
            assert all([x in f for x in ['env', 'metadata', 'policy', 'misc']])
            
            action_T = f['policy']['action'].shape[0]
            self._ncam = f['env'].attrs.get('ncam', 0)
            assert self._ncam > 0, "must be at least 1 camera!"
            img_T = min([len(f['env']['cam{}_video'.format(i)]) for i in range(self._ncam)])

            assert action_T >= self._hparams.context_frames + self._hparams.predicted_frames - 1, "not enough actions per traj!"
            assert img_T >= self._hparams.predicted_frames + self._hparams.context_frames, "not enough images per traj!"

            state_t = f['env']['state'].shape[0]
            if state_t >= self._hparams.predicted_frames + self._hparams.context_frames:
                self._valid_keys = ['actions', 'images', 'state']
            else:
                self._valid_keys['actions', 'images']
        
        self._init_queues(hdf5_files)
    
    def _init_queues(self, hdf5_files):
        import pdb; pdb.set_trace()
        print()


    def _get_default_hparams(self):
        default_params = super(HDF5VideoDataset, self)._get_default_hparams()
        
        # set None if you want a random seed for dataset shuffling
        default_params.add_hparam('RNG', 11381294392481135266)
        default_params.add_hparam('splits', [0.9, 0.05, 0.05])   # (train, val, test) split
        
        default_params.add_hparam('context_frames', 2)
        default_params.add_hparam('predicted_frames', 13)
        default_params.add_hparam('max_start', -1)
        
        return default_params


if __name__ == '__main__':
    path = '/home/sudeep/Desktop/test_dataset/test0'
    batch_size = 1
    dataset = HDF5VideoDataset(path, batch_size)

    images, actions = dataset['images'], dataset['actions']
    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    for _ in range(10):
        print('batch')
        sess.run([images, actions])
