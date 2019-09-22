import os
import numpy as np
import tensorflow as tf
import pickle as pkl
from collections import OrderedDict
import h5py


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def save_tf_record(filename, trajectory_list, sequence_manifest, metadata_manifest):
    """
    saves data_files from one sample trajectory into one tf-record file
    """

    def check_against_manifest(features, manifest):
        if manifest is None and features is not None:
            raise ValueError("Manifest is none, but values were given. Maybe you didn't set manifest?")
        if features is None and manifest is not None:
            raise ValueError("Feature is none, but manifest is given. Maybe you didn't pass in features?")

        for k in features.keys():
            assert k in manifest, "Key {} passed to writer but not in manifest".format(k)
        for k in manifest.keys():
            assert k in features, "Key {} in manifest but not in given record".format(k)

    filename = filename + '.tfrecords'
    print(filename)
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(filename, options=options)

    for meta_data, sequence_data in trajectory_list:
        check_against_manifest(meta_data, metadata_manifest)

        feature = {}
        for tind, feats in enumerate(sequence_data):
            check_against_manifest(feats, sequence_manifest)
            for k in feats:
                feature['{}/{}'.format(tind, k)] = feats[k]
        for k in meta_data:
            feature[k] = meta_data[k]

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()


class RecordSaver:
    def __init__(self, data_save_dir, sequence_length=None, traj_per_file=1, offset=0, split=(0.90, 0.05, 0.05)):
        self._traj_buffers = [[] for _ in range(3)]
        self._save_counters = [0 for _ in range(3)]

        dirs_to_create = ['train', 'test', 'val']

        dirs_to_create = ['{}/{}'.format(data_save_dir, d) for d in dirs_to_create]
        for d in dirs_to_create:
            if not os.path.exists(d):
                print('Creating dir:', d)
                os.makedirs(d)


        self._base_dir = data_save_dir
        self._train_test_val = split
        self._traj_per_file = traj_per_file
        self._metadata_keys, self._sequence_keys, self._T = None, None, sequence_length
        self._offset = offset

        self._force_draw = False
        if any([i == 1 for i in split]):
            print('Forcing Draw')
            self._force_draw = True

    def add_traj(self, traj):
        draw = None
        if not self._force_draw:
            for i in range(len(self._traj_buffers)):
                if self._save_counters[i] == 0 and self._train_test_val[i] > 0 and np.random.randint(0, 2) == 1:
                    draw = i
                    continue

        if draw is None:
            draw = np.random.choice([0, 1, 2], 1, p=self._train_test_val)[0]

        self._traj_buffers[draw].append(traj)
        self._save()

    def flush(self):
        self._save(True)

    def add_metadata_entry(self, key, shape, type):
        assert type in ['Float', 'Int', 'Byte'], 'Given type: {} is invalid'.format(type)
        if self._metadata_keys is None:
            self._metadata_keys = OrderedDict()
        self._metadata_keys[key] = (shape, type)

    @property
    def sequence_length(self):
        return self._T

    @sequence_length.setter
    def sequence_length(self, T):
        self._T = T

    def add_sequence_entry(self, key, shape, type):
        if self._T is None:
            raise ValueError("sequence_length not set during construction!")

        assert type in ['Float', 'Int', 'Byte'], 'Given type: {} is invalid'.format(type)
        if self._sequence_keys is None:
            self._sequence_keys = OrderedDict()
        self._sequence_keys[key] = (shape, type)

    def save_manifest(self):
        if self._metadata_keys is None and self._sequence_keys is None:
            raise ValueError("Keys never added to manifest")

        with open('{}/manifest.txt'.format(self._base_dir), 'w') as f:
            f.write('# DATA MANIFEST\n')
            f.write('##############################################################\n\n')
            if self._metadata_keys is not None:
                f.write('# Trajectory meta-data\n')
                for key in self._metadata_keys:
                    shape, feat_dtype = self._metadata_keys[key]
                    shape_str = ''

                    for s in shape:
                        shape_str += ' {},'.format(s)

                    f.write('{}: ({}) - {}\n'.format(key, shape_str[1:-1], feat_dtype))
                f.write('\n##############################################################\n\n')

            if self._sequence_keys is not None:
                f.write('# Sequence Data\n')
                f.write('Timesteps: {}\n'.format(self._T))
                for key in self._sequence_keys:
                    shape, feat_dtype = self._sequence_keys[key]
                    shape_str = ''

                    for s in shape:
                        shape_str += ' {},'.format(s)

                    f.write('{}: ({}) - {}\n'.format(key, shape_str[1:-1], feat_dtype))

        with open('{}/manifest.pkl'.format(self._base_dir), 'wb') as f:
            manifest_dict = {}
            manifest_dict['sequence_data'] = self._sequence_keys
            manifest_dict['traj_metadata'] = self._metadata_keys
            manifest_dict['T'] = self._T
            pkl.dump(manifest_dict, f)

    def __len__(self):
        return sum(self._save_counters)

    def _save(self, flush = False):
        for i, name in zip(range(3), ['train', 'test', 'val']):
            buffer = self._traj_buffers[i]
            if len(buffer) == 0:
                continue
            elif flush or len(buffer) % self._traj_per_file == 0:
                next_counter = self._save_counters[i] + len(buffer)
                
                num_saved = sum(self._save_counters) + self._offset
                next_total = num_saved + len(buffer)

                folder = '{}/{}'.format(self._base_dir, name)
                file = '{}/traj_{}_to_{}'.format(folder, num_saved, next_total - 1)
                save_tf_record(file, buffer, self._sequence_keys, self._metadata_keys)

                self._traj_buffers[i] = []
                self._save_counters[i] = next_counter


class HDF5SaverBase():
    def __init__(self, save_dir, traj_per_file,
                 offset=0, split=(0.90, 0.05, 0.05), split_train_val_test=True):

        self.train_test_val_split = split
        self.split_train_val_test = split_train_val_test
        self.traj_per_file = traj_per_file
        self.traj_lists = [[], [], []]   # train val test lists
        self.save_dir = save_dir
        self.traj_count = offset
        # TODO make this create dataset_spec

    def save_hdf5(self, traj_list, prefix):
        savedir = self.save_dir + '/hdf5/{}'.format(prefix) if self.split_train_val_test else self.save_dir + '/hdf5'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        self.traj_count += 1

        with h5py.File(savedir + '/traj_{}to{}'.format((self.traj_count  - 1)*self.traj_per_file,
                                                self.traj_count*self.traj_per_file) + '.h5', 'w') as F:

            F['traj_per_file'] = self.traj_per_file
            for i, traj in enumerate(traj_list):
                key = 'traj{}'.format(i)

                assert traj.images.dtype == np.uint8, 'image need to be uint8!'
                for name, value in traj.items():
                    F[key + '/' + name] = value

    def make_traj(self):
        raise NotImplementedError()

    def save_traj(self):
        raise NotImplementedError()
        
    def _save_traj(self, traj):
        draw = np.random.choice([0, 1, 2], 1, p=self.train_test_val_split)[0]
        self.traj_lists[draw].append(traj)

        for i, prefix in enumerate(['train', 'val', 'test']):
            if len(self.traj_lists[i]) == self.traj_per_file:
                self.save_hdf5(self.traj_lists[i], prefix)
                self.traj_lists[i] = []

    def make_dataset(self):
        boundaries = np.cumsum(np.array(self.train_test_val_split) * len(self.filenames), 0).astype(int)
    
        self.make_phase(self.filenames[:boundaries[0]], 'train')
    
        self.make_phase(self.filenames[boundaries[0]:boundaries[1]], 'val')
    
        self.make_phase(self.filenames[boundaries[1]:], 'test')