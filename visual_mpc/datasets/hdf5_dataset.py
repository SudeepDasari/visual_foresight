from visual_mpc.datasets.base_dataset import BaseVideoDataset
import h5py
import tensorflow as tf
import numpy as np
import cv2
import glob
import random
import math
from collections import OrderedDict


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

            self._action_T, self._adim = f['policy']['actions'].shape

            self._ncam = f['env'].attrs.get('n_cams', 0)
            assert self._ncam > 0, "must be at least 1 camera!"
            self._img_T = min([len(f['env']['cam{}_video'.format(i)]) for i in range(self._ncam)])
            self._img_dim = f['env']['cam0_video']['frame0'].attrs['shape'][:2]

            assert self._action_T >= self._hparams.context_frames + self._hparams.predicted_frames - 1, "not enough actions per traj!"
            assert self._img_T >= self._hparams.predicted_frames + self._hparams.context_frames, "not enough images per traj!"

            self._state_T, self._sdim = f['env']['state'].shape
            if self._state_T >= self._hparams.predicted_frames + self._hparams.context_frames:
                self._valid_keys = ['actions', 'images', 'state']
                self._parser_dtypes = [tf.float32, tf.string, tf.float32]
            else:
                self._valid_keys = ['actions', 'images']
                self._parser_dtypes = [tf.float32, tf.string]
        
        self._init_queues(dataset_contents)
    
    def _read_hdf5(self, filename):
        with h5py.File(filename, 'r') as hf:
            image_strs = []
            for t in range(self._img_T):
                for n in range(self._ncam):
                    image_strs.append(hf['env']['cam{}_video'.format(n)]['frame{}'.format(t)][:].tostring())
            actions = hf['policy']['actions'][:].astype(np.float32)

            if 'state' in self._valid_keys:
                return actions, image_strs, hf['env']['state'][:].astype(np.float32)
        return actions, image_strs
    
    def _decode_act_img_state(self, actions, image_strs, states):
        out_dict = self._decode_act_img(actions, image_strs)
        out_dict['states'] = tf.reshape(states, [self._state_T, self._sdim])
        return out_dict
    
    def _decode_act_img(self, actions, image_strs):
        actions = tf.reshape(actions, [self._action_T, self._adim])
        frames = []
        i = 0
        for t in range(self._img_T):
            for n in range(self._ncam):
                img_decoded = tf.image.decode_jpeg(image_strs[i], channels=3)
                frames.append(tf.reshape(img_decoded, [self._img_dim[0], self._img_dim[1], 3]))
                i += 1
        
        method = tf.image.ResizeMethod.BILINEAR
        if any([self._hparams.img_dims[i] < self._img_dim[i] for i in range(2)]):
            method = tf.image.ResizeMethod.AREA
        
        frames = tf.image.resize_images(frames, self._hparams.img_dims, method=method)
        frames = tf.reshape(frames, [self._img_T, self._ncam, self._hparams.img_dims[0], self._hparams.img_dims[1], 3])
        return {'actions': actions, 'images': frames}
    
    def _init_queues(self, hdf5_files):
        assert len(self.MODES) == len(self._hparams.splits), "len(splits) should be the same as number of MODES!"
        split_lengths = [int(math.ceil(len(hdf5_files) * x)) for x in self._hparams.splits[1:]]
        split_lengths = np.cumsum([0, len(hdf5_files) - sum(split_lengths)] + split_lengths)
        splits = OrderedDict()
        for i, name in enumerate(self.MODES):
            splits[name] = hdf5_files[split_lengths[i]:split_lengths[i+1]]

        self._mode_datasets = {}
        for name, files in splits.items():
            dataset = tf.data.Dataset.from_tensor_slices(files)
            dataset = dataset.repeat(self._hparams.num_epochs)
            dataset = dataset.map(
                                    lambda filename: tuple(tf.py_func(
                                    self._read_hdf5, [filename], self._parser_dtypes))
                                )
            if 'state' in self._valid_keys:
                dataset = dataset.map(self._decode_act_img_state)
            else:
                dataset = dataset.map(self._decode_act_img)
            
            dataset = dataset.shuffle(self._hparams.buffer_size)
            dataset = dataset.batch(self._batch_size)
            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()

            output_element = {}
            for k in list(next_element.keys()):
                output_element[k] = tf.reshape(next_element[k],
                                               [self._batch_size] + next_element[k].get_shape().as_list()[1:])
            
            self._mode_datasets[name] = output_element

    def _get(self, key, mode):
        assert key in self._mode_datasets[mode], "Key {} is not recognized for mode {}".format(key, mode)

        return self._mode_datasets[mode][key]
    
    def _get_default_hparams(self):
        default_params = super(HDF5VideoDataset, self)._get_default_hparams()
        
        # set None if you want a random seed for dataset shuffling
        default_params.add_hparam('RNG', 11381294392481135266)
        default_params.add_hparam('splits', [0.9, 0.05, 0.05])   # (train, val, test) split
        default_params.add_hparam('img_dims', (48, 64))
        default_params.add_hparam('context_frames', 2)
        default_params.add_hparam('predicted_frames', 13)
        default_params.add_hparam('max_start', -1)
        
        return default_params


if __name__ == '__main__':
    import moviepy.editor as mpy
    
    path = '/home/sudeep/Desktop/test_dataset/test0'
    batch_size = 1
    # small shuffle buffer for testing
    dataset = HDF5VideoDataset(path, batch_size, hparams_dict={'buffer_size':10})

    images, actions = dataset['images'], dataset['actions']
    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    imgs, acts = sess.run([images[0], actions[0]])
    for i in range(imgs.shape[1]):
        mpy.ImageSequenceClip([fr for fr in imgs[:, i]], fps=5).write_gif('test{}.gif'.format(i))
    print(acts)
