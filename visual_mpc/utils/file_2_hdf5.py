from visual_mpc.datasets.save_util.hdf5_saver import save_hdf5
import argparse
import glob
import json
import cv2
import numpy as np
import os
import sys
if sys.version_info[0] == 2:
    import cPickle as pkl
else:
    import pickle as pkl


def safe_open_pkl(path):
    if os.path.exists(path):
        if sys.version_info[0] == 2:
            return pkl.load(open(path, 'rb'))
        return pkl.load(open(path, 'rb'), encoding='latin1')
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="converts dataset from pkl format to hdf5")
    parser.add_argument('input_folders', type=str, help='which files to load (seperated by commas)')
    parser.add_argument('output_folder', type=str, help='where to save')
    parser.add_argument('--add_goal_flag', action='store_true', default=False, help="whether or not to add goal_reached flag to misc")
    parser.add_argument('--counter', type=int, help='where to start counter', default=0)
    args = parser.parse_args()

    datasets = []
    for f in args.input_folders.split(','):
        datasets += glob.glob('{}/*/*/*/traj_group*'.format(os.path.expanduser(f)))

    counter = args.counter
    for d in datasets:
        meta_data = json.load(open('{}/hparams.json'.format(d), 'r'))
        print('on dataset: {}'.format(d))
        d_save = 0
        trajs = glob.glob('{}/traj*'.format(d))
        has_cam, dataset_ncams, dataset_T, dataset_im_dim = None, None, None, None

        for t in trajs:
            env_obs = safe_open_pkl('{}/obs_dict.pkl'.format(t))
            policy_out = safe_open_pkl('{}/policy_out.pkl'.format(t))
            agent_data = safe_open_pkl('{}/agent_data.pkl'.format(t))

            if not env_obs or not policy_out or not agent_data:
                continue
            
            n_cams = len(glob.glob('{}/images*'.format(t)))
            if n_cams:
                T = min([len(glob.glob('{}/images{}/*.jpg'.format(t, i))) for i in range(n_cams)])
                height, width = cv2.imread('{}/images0/im_0.jpg'.format(t)).shape[:2]

                if d_save == 0:
                    has_cam, dataset_ncams, dataset_T, dataset_im_dim = True, n_cams, T, (height, width)
                    print('first traj has {} cameras, {} timesteps per cam, {} dim'.format(n_cams, T, dataset_im_dim))
                elif n_cams != dataset_ncams or dataset_T != T or dataset_im_dim != (height, width):
                    continue

                env_obs['images'] = np.zeros((T, n_cams, height, width, 3), dtype=np.uint8)

                cam_ok = True
                for n in range(n_cams):
                    for time in range(T):
                        im_path = '{}/images{}/im_{}.jpg'.format(t, n, time)
                        if not os.path.exists(im_path):
                            cam_ok = False
                            continue
                        env_obs['images'][time, n] = cv2.imread(im_path)
                if not cam_ok:
                    continue
            elif d_save == 0:
                print("first traj {} has no cameras, please check!".format(t))
                has_cam = False
            elif has_cam:
                continue

            if args.add_goal_flag:
                good_states = np.logical_and(env_obs['state'][:-1, 2] >= 0.9,  env_obs['state'][:-1, -1] > 0)
                agent_data['goal_reached'] = np.sum(np.logical_and(env_obs['finger_sensors'][:-1, 0] > 0, good_states)) >= 2
            
            save_hdf5('{}/traj{}.hdf5'.format(args.output_folder, counter), env_obs, policy_out, agent_data, meta_data)
            counter += 1
            d_save += 1
        print('saved {} trajs'.format(d_save))

    print('created {} output trajs!'.format(counter - args.counter))
