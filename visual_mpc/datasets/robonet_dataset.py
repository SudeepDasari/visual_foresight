from visual_mpc.datasets.base_dataset import BaseVideoDataset
from visual_mpc.datasets.hdf5_dataset import HDF5VideoDataset
from visual_mpc.datasets.save_util.filter_dataset import cached_filter_hdf5
import glob
import copy


class RoboNetDataset(BaseVideoDataset):
    def __init__(self, directory, batch_size, hparams_dict=dict()):
        hdf5_params = copy.deepcopy(hparams_dict)
        self._filters = hdf5_params.pop('filters', [])
        self._source_views = hdf5_params.pop('source_views', [])
        super(RoboNetDataset, self).__init__(directory, batch_size, hdf5_params)
    
    @staticmethod
    def _get_default_hparams():
        # allows us to get access to default params of hdf5 dataset in self._hparams
        return HDF5VideoDataset._get_default_hparams()

    def _init_dataset(self):
        dataset_files = glob.glob('{}/*.hdf5'.format(self._base_dir))
        assert len(dataset_files) > 0, "couldn't find dataset at {}".format(self._base_dir)

        filtered_datasets = cached_filter_hdf5(dataset_files, '{}/filter_cache.pkl'.format(self._base_dir))
        for k in filtered_datasets.keys():
            print('calib - {}, insert - {}, len - {}'.format(k['metadata/camera_calibration'], k['metadata/bin_insert'], len(filtered_datasets[k])))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="converts dataset from pkl format to hdf5")
    parser.add_argument('input_folder', type=str, help='folder containing hdf5 files')
    args = parser.parse_args()

    path = args.input_folder
    rn = RoboNetDataset(path, 1, {'filters': ["asdf=2,aad"], 'source_views':[1,2,3]})