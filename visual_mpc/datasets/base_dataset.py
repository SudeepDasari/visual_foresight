import os
from tensorflow.contrib.training import HParams


class BaseVideoDataset(object):
    MODES = ['train', 'test', 'val']

    def __init__(self, directory, batch_size, hparams_dict=dict()):
        if not os.path.exists(directory):
            raise FileNotFoundError('Base directory {} does not exist'.format(directory))

        self._base_dir = directory
        self._batch_size = batch_size

        # read dataset manifest and initialize hparams
        self._hparams = self._get_default_hparams().override_from_dict(hparams_dict)
        
        #initialize dataset class
        self._init_dataset()

    def _init_dataset(self):
        raise NotImplementedError

    def _get_default_hparams(self):
        default_dict = {'shuffle': True,
                        'num_epochs': None,
                        'buffer_size': 512
                        }
        return HParams(**default_dict)

    
    def get(self, key, mode='train'):
        if mode not in self.MODES:
            raise ValueError('Mode {} not valid! Dataset has following modes: {}'.format(mode, self.MODES))
        return self._get(key, mode)
    
    def _get(self, key, mode):
        raise NotImplementedError

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if len(item) != 2:
                raise KeyError('Index should be in format: [Key, Mode] or [Key] (assumes default train mode)')
            key, mode = item
            return self.get(key, mode)

        return self.get(item)
