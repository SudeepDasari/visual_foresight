from visual_mpc.datasets.save_util.record_saver import RecordSaver, float_feature, bytes_feature, int64_feature
import numpy as np
import os


"""
Probably should look into a better of way of doing this.
Having two functions for getting type/serializing seems like a waste....
"""


def get_dtype(datum):
    if isinstance(datum, int):
        return 'Int'
    elif isinstance(datum, float):
        return 'Float'
    elif isinstance(datum, bool):
        return 'Int'
    elif isinstance(datum, np.ndarray):
        if datum.dtype == np.uint8:
            return 'Byte'
        elif datum.dtype.kind == 'i':
            return 'Int'
        elif datum.dtype.kind == 'f':
            return 'Float'
    raise ValueError('datum {} has unknown dtype'.format(datum))


def convert_datum(datum):
    if isinstance(datum, np.ndarray):
        if datum.dtype == np.uint8:
            return bytes_feature(datum.tostring())
        elif datum.dtype.kind == 'i':
            return int64_feature(datum.flatten().tolist())
        elif datum.dtype.kind == 'f':
            return float_feature(datum.flatten().tolist())
    elif isinstance(datum, float):
        return float_feature([datum])
    elif isinstance(datum, int):
        return int64_feature([datum])
    elif isinstance(datum, bool):
        return int64_feature([int(datum)])

    raise ValueError('datum {} has unknown dtype'.format(datum))


class GeneralAgentSaver:
    """
    Serializes trajectory data and sends to RecordSaver to store as TFRecord
    """
    def __init__(self, save_dir, sequence_length, seperate_good=False, traj_per_file=128, offset=0, split=(0.90, 0.05, 0.05)):
        self._base_dir = save_dir
        self._seperate_good = seperate_good
        self._manifest_saved, self._T = False, sequence_length

        if seperate_good:
            self._good_saver = RecordSaver('{}/good'.format(self._base_dir), sequence_length, traj_per_file, offset, split)
            self._bad_saver = RecordSaver('{}/bad'.format(self._base_dir), sequence_length, traj_per_file, offset, split)
        else:
            self._saver = RecordSaver(self._base_dir, sequence_length, traj_per_file, offset, split)

    def _save_manifests(self, agent_data, obs, policy_out):
        def get_shape(datum):
            if isinstance(datum, np.ndarray):
                return datum.shape
            return tuple([1])

        if self._seperate_good:
            savers = [self._good_saver, self._bad_saver]
        else:
            savers = [self._saver]
        for s in savers:
            if agent_data is not None:
                for k in agent_data:
                    s.add_metadata_entry(k, get_shape(agent_data[k]), get_dtype(agent_data[k]))
            if obs is not None:
                for k in obs:
                    if k == 'images':
                        ncam = obs[k].shape[1]
                        for c in range(ncam):
                            s.add_sequence_entry('env/image_view{}/encoded'.format(c),
                                                 get_shape(obs[k][0, 0]), get_dtype(obs[k][0, 0]))
                    else:
                        key_name = 'env/{}'.format(k)
                        s.add_sequence_entry(key_name, get_shape(obs[k][0]), get_dtype(obs[k][0]))
            if policy_out is not None and len(policy_out) > 0:
                for k in policy_out[0]:
                    key_name = 'policy/{}'.format(k)
                    s.add_sequence_entry(key_name, get_shape(policy_out[0][k]), get_dtype(policy_out[0][k]))
            s.save_manifest()

    def save_traj(self, agent_data, obs, policy_out):
        is_good = None
        if self._seperate_good:
            is_good = agent_data.pop('goal_reached')

        if 'traj_ok' in agent_data and not agent_data.pop('traj_ok'):
            print('RECEIVED NOT OKAY TRAJ, MAYBE UP ITERS?')
            return

        if not self._manifest_saved:
            self._save_manifests(agent_data, obs, policy_out)
            self._manifest_saved = True

        sequence_data = []
        meta_data_dict = {}

        for k in agent_data:
            meta_data_dict[k] = convert_datum(agent_data[k])

        for t in range(self._T):
            step_dict = {}
            for k in obs:
                if k == 'images':
                    ncam = obs[k].shape[1]
                    for c in range(ncam):
                        step_dict['env/image_view{}/encoded'.format(c)] = convert_datum(obs[k][t, c])
                else:
                    step_dict['env/{}'.format(k)] = convert_datum(obs[k][t])
            if len(policy_out) > t:
                for k in policy_out[t]:
                    step_dict['policy/{}'.format(k)] = convert_datum(policy_out[t][k])

            sequence_data.append(step_dict)

        traj = (meta_data_dict, sequence_data)

        if self._seperate_good and is_good:
            self._good_saver.add_traj(traj)
        elif self._seperate_good:
            self._bad_saver.add_traj(traj)
        else:
            self._saver.add_traj(traj)

    def flush(self):
        if self._seperate_good:
            self._good_saver.flush()
            self._bad_saver.flush()
            total = len(self._bad_saver) + len(self._good_saver)
            if total > 0:
                print('Perc good: {}'.format(len(self._good_saver) / float(total) * 100.))
        else:
            self._saver.flush()


def record_worker(queue, save_dir, sequence_length, seperate_good, traj_per_file, offset=0, split=(0.90, 0.05, 0.05)):
    print('started saver with PID:', os.getpid())
    print('saving to {}'.format(save_dir))
    saver = GeneralAgentSaver(save_dir, sequence_length, seperate_good, traj_per_file, offset, split)
    data = queue.get(True)
    counter = 0
    while data is not None:
        counter += 1
        agent_data, obs, policy_out = data
        saver.save_traj(agent_data, obs, policy_out)
        data = queue.get(True)
    print('Saved {} as tfrecords'.format(counter))
    saver.flush()