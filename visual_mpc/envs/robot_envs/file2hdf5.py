import h5py
import cv2
import numpy as np
import copy
import os
import imageio
import io
from multiprocessing import Pool, Manager
from visual_mpc.utils.sync import ManagedSyncCounter
import random
import functools
from tqdm import tqdm


MANDATORY_KEYS = ['camera_configuration', 'policy_desc', 'bin_type', 'bin_insert', 'contains_annotation',
                                'robot', 'gripper', 'background', 'action_space', 'object_classes', 'primitives', 'camera_type']


def serialize_image(img):
    assert img.dtype == np.uint8, "Must be uint8!"
    return cv2.imencode('.jpg', img)[1]


def serialize_video(imgs, temp_name_append):
    mp4_name = './temp{}.mp4'.format(temp_name_append)
    try:
        assert imgs.dtype == np.uint8, "Must be uint8 array!"
        assert not os.path.exists(mp4_name), "file {} exists!".format(mp4_name)
        # this is a hack to ensure imageio succesfully saves as a mp4 (instead of getting encoding confused)
        writer = imageio.get_writer(mp4_name)
        [writer.append_data(i[:, :, ::-1]) for i in imgs]
        writer.close()

        f = open(mp4_name, 'rb')
        buf = f.read()
        f.close()
    finally:
        if os.path.exists(mp4_name):
            os.remove(mp4_name)

    return np.frombuffer(buf, dtype=np.uint8)


def save_dict(data_container, dict_group, video_encoding, t_index):
    for k, d in data_container.items():
                    if 'images' == k:
                        T, n_cams = d.shape[:2]
                        dict_group.attrs['n_cams'] = n_cams

                        for n in range(n_cams):
                            cam_group = dict_group.create_group("cam{}_video".format(n))
                            if video_encoding == 'mp4':
                                data = cam_group.create_dataset("frames", data=serialize_video(d[:, n], t_index))
                                data.attrs['shape'] = d[0, n].shape
                                data.attrs['T'] = d.shape[0]
                                data.attrs['image_format'] = 'RGB'
                            elif video_encoding == 'jpeg':
                                for t in range(T):
                                    data = cam_group.create_dataset("frame{}".format(t), data=serialize_image(d[t, n]))
                                    data.attrs['shape'] = d[t, n].shape
                            else:
                                raise ValueError
                    elif 'image' in k:
                        data = dict_group.create_dataset(k, data=serialize_image(d))
                        data.attrs['shape'] = d.shape
                    else:
                        dict_group.create_dataset(k, data=d)


def save_hdf5(filename, env_obs, policy_out, agent_data, meta_data, video_encoding='mp4', t_index=None):
    if t_index is None:
        t_index = random.randint(0, 9999999)
    # meta-data includes calibration "number", policy "type" descriptor, environment bounds
    with h5py.File(filename, 'w') as f:
        [save_dict(data_container, f.create_group(name), video_encoding, t_index) for data_container, name in zip([env_obs, agent_data], ['env', 'misc'])]

        policy_dict = {}
        first_keys = list(policy_out[0].keys())
        for k in first_keys:
            assert all([k in p for p in policy_out[1:]]), "hdf5 format requires keys must be uniform across time!"
            policy_dict[k] = np.concatenate([p[k][None] for p in policy_out], axis=0)
        save_dict(policy_dict, f.create_group('policy'), video_encoding, t_index)

        meta_data_group = f.create_group('metadata')
        for mandatory_key in MANDATORY_KEYS:
            meta_data_group.attrs[mandatory_key] = meta_data.pop(mandatory_key)
        
        for k in meta_data.keys():
            meta_data_group.attrs[k] = meta_data[k]
            

def save_worker(traj_data, cntr, group_name=''):
    t_index = random.randint(0, 9999999)
    t, meta_data = traj_data

    try:
        env_obs = pkl.load(open('{}/obs_dict.pkl'.format(t), 'rb'), encoding='latin1')
        if meta_data['contains_annotation']:
            env_obs['bbox_annotations'] = pkl.load(open('{}/annotation_array.pkl'.format(t), 'rb'), encoding='latin1')
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

        def store_in_metadata_if_exists(key):  
            if key in agent_data:
                meta_data[key] = agent_data.pop(key)
        [store_in_metadata_if_exists(k) for k in ['goal_reached', 'term_t']]

        c = cntr.ret_increment
        save_hdf5('{}/{}traj{}.hdf5'.format(args.output_folder, group_name, c), env_obs, policy_out, agent_data, meta_data, video_encoding, t_index)
        return True
    except (FileNotFoundError, NotADirectoryError):
        return False


if __name__ == '__main__':
    import argparse
    import glob
    import json
    import random
    import sys
    import os
    import shutil
    import math
    if sys.version_info[0] == 2:
        import cPickle as pkl
        input_fn = raw_input
    else:
        import pickle as pkl
        input_fn = input

    parser = argparse.ArgumentParser(description="converts dataset from pkl format to hdf5")
    parser.add_argument('input_folder', type=str, help='where raw files are stored')
    parser.add_argument('output_folder', type=str, help='where to save')
    parser.add_argument('--output_group_name', type=str, default='', help='name to prepend in front of trajs')
    parser.add_argument('--video_jpeg_encoding', action='store_true', default=False, help='uses jpeg encoding for video frames instead of mp4')
    parser.add_argument('--counter', type=int, help='where to start counter', default=0)
    parser.add_argument('--n_workers', type=int, help='number of multi-threaded workers', default=1)
    args = parser.parse_args()

    assert args.n_workers >= 1, "can't have less than 1 worker thread!"
    args.input_folder, args.output_folder = [os.path.expanduser(x) for x in (args.input_folder, args.output_folder)]
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    elif input_fn('path {} exists, should folder be deleted? (y/n): '.format(args.output_folder)) == 'y':
        shutil.rmtree(args.output_folder)
        os.makedirs(args.output_folder)   
    
    if args.video_jpeg_encoding:
        video_encoding = 'jpeg'
    else:
        video_encoding = 'mp4'
        if len(glob.glob('./temp*.mp4')) != 0:
            print("Please delete all temp*.mp4 files! (needed for saving)")
            raise EnvironmentError
    
    traj_groups = glob.glob(args.input_folder + "*/*/*/*")
    print('found {} traj groups!'.format(len(traj_groups)))

    trajs, annotations_loaded = [], 0
    for group in traj_groups:
        meta_data_dict = json.load(open('{}/hparams.json'.format(group), 'r'))
        group_trajs = glob.glob('{}/*'.format(group))
        for t in group_trajs:
            traj_meta_data = copy.deepcopy(meta_data_dict)
            traj_meta_data['object_batch'] = group
            if os.path.exists('{}/annotation_array.pkl'.format(t)):
                traj_meta_data['contains_annotation'] = True
                annotations_loaded += 1
            else:
                traj_meta_data['contains_annotation'] = False
            
            if isinstance(traj_meta_data['object_classes'], str):
                traj_meta_data['object_classes'] = traj_meta_data['object_classes'].split("+")
            
            assert all([k in traj_meta_data for k in MANDATORY_KEYS]), 'metadata for {} is missing keys!'.format(t)
            assert isinstance(traj_meta_data['object_classes'], list), "did not split object classes!"
            assert all([isinstance(x, str) for x in traj_meta_data['object_classes']]), 'object classes is not a string!'

            trajs.append((t, traj_meta_data))
    random.shuffle(trajs)
    
    print('Loaded {} trajectories with {} annotations!'.format(len(trajs), annotations_loaded))

    cntr = ManagedSyncCounter(Manager(), args.counter)
    if args.n_workers == 1:
        saved = 0
        for t in tqdm(trajs):
            saved += save_worker(t, cntr, args.output_group_name)
        
        print('saved {} total trajs'.format(saved))
    else:
        map_fn = functools.partial(save_worker, cntr=cntr, group_name=args.output_group_name)
        p = Pool(args.n_workers)
        print('saved {} total trajs'.format(sum(tqdm(p.imap_unordered(map_fn, trajs), total=len(trajs)))))
