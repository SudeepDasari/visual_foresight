import h5py
import cv2
import numpy as np
import copy


def serialize_image(img):
    assert img.dtype == np.uint8, "Must be uint8!"
    return cv2.imencode('.jpg', img)[1]


def save_dict(data_container, dict_group):
    for k, d in data_container.items():
                    if 'images' in k:
                        T, n_cams = d.shape[:2]
                        for n in range(n_cams):
                            cam_group = dict_group.create_group("image{}".format(n))
                            for t in range(T):
                                data = cam_group.create_dataset("frame{}".format(t), data=serialize_image(d[t, n]))
                                data.attrs['shape'] = d[t, n].shape
                    elif 'image' in k:
                        data = dict_group.create_dataset(k, data=serialize_image(d))
                        data.attrs['shape'] = d.shape
                    else:
                        dict_group.create_dataset(k, data=d)


def save_hdf5(filename, env_obs, policy_out, agent_data, meta_data=None):
    # meta-data includes calibration "number", policy "type" descriptor, environment bounds
    with h5py.File(filename, 'w') as f:
        [save_dict(data_container, f.create_group(name)) for data_container, name in zip([env_obs, agent_data], ['env', 'misc'])]

        policy_dict = {}
        first_keys = list(policy_out[0].keys())
        for k in first_keys:
            assert all([k in p for p in policy_out[1:]]), "hdf5 format requires keys must be uniform across time!"
            policy_dict[k] = np.concatenate([p[k][None] for p in policy_out], axis=0)
        save_dict(policy_dict, f.create_group('policy'))

        if meta_data is not None and type(meta_data) == dict:
            meta_data = copy.deepcopy(meta_data)

            meta_data_group = f.create_group('metadata')
            for mandatory_key in ['calib_number', 'policy_desc', 'bounds']:
                meta_data_group.attrs[mandatory_key] = meta_data.pop(mandatory_key)
            
            for k in meta_data.keys():
                meta_data_group.attrs[k] = meta_data[k]
            

if __name__ == '__main__':
    import argparse
    import glob
    import json
    import random
    import sys
    import os

    if sys.version_info[0] == 2:
        import cPickle as pkl
    else:
        import pickle as pkl

    parser = argparse.ArgumentParser(description="converts dataset from pkl format to hdf5")
    parser.add_argument('input_folder', type=str, help='which files to load')
    parser.add_argument('output_folder', type=str, help='where to save')
    parser.add_argument('--meta_data_path', type=str, help='where to get metadata json', default='')
    parser.add_argument('--counter', type=int, help='where to start counter', default=0)
    args = parser.parse_args()

    trajs = glob.glob(args.input_folder + "/traj*")
    random.shuffle(trajs)

    meta_data = None
    if args.meta_data_path:
        meta_data = json.load(open(args.meta_data_path))
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    cntr = args.counter

    for t in trajs:
        env_obs = pkl.load(open('{}/obs_dict.pkl'.format(t), 'rb'), encoding='latin1')
        n_cams = len(glob.glob('{}/images*'.format(t)))
        if n_cams:
            T = min([len(glob.glob('{}/images{}/*.jpg'.format(t, i))) for i in range(n_cams)])
            height, width = cv2.imread('{}/images0/im_0.jpg'.format(t)).shape[:2]
            env_obs['images'] = np.zeros((T, n_cams, height, width, 3), dtype=np.uint8)

            for n in range(n_cams):
                for time in range(T):
                    env_obs['images'][time, n] = cv2.imread('{}/images{}/im_{}.jpg'.format(t, n, time))

        policy_out = pkl.load(open('{}/policy_out.pkl'.format(t), 'rb'), encoding='latin1')
        agent_data = pkl.load(open('{}/agent_data.pkl'.format(t), 'rb'), encoding='latin1')

        save_hdf5('{}/traj{}.hdf5'.format(args.output_folder, cntr), env_obs, policy_out, agent_data, meta_data)
        cntr += 1
