import argparse
from multiprocessing import Pool, Process, Manager
from visual_mpc.agent.utils.traj_saver import record_worker
import os
import cv2
import cPickle as pkl
import numpy as np
import glob
import random
import copy


def save_worker(save_conf):
    assigned_files, record_queue, T, target_width, seperate, infer_gripper, separate_views = save_conf
    target_dim = None
    ncam = None
    for traj in assigned_files:
        if target_dim == None:
            ncam = len(glob.glob('{}/images*/'.format(traj)))
            img = cv2.imread('{}/images0/im_0.jpg'.format(traj))
            old_dim = img.shape[:2]
            resize_ratio = target_width / float(old_dim[1])
            target_dim = (target_width, int(old_dim[0] * resize_ratio))
            print('resizing to {}'.format(target_dim[::-1]))

        agent_data = pkl.load(open('{}/agent_data.pkl'.format(traj), 'rb'))
        obs_dict = pkl.load(open('{}/obs_dict.pkl'.format(traj), 'rb'))
        policy_out = pkl.load(open('{}/policy_out.pkl'.format(traj), 'rb'))

        imgs = np.zeros((T, ncam, target_dim[1], target_dim[0], 3), dtype = np.uint8)

        for t in range(T):
            for n in range(ncam):
                img = cv2.imread('{}/images{}/im_{}.jpg'.format(traj, n, t))[:, :, ::-1]
                if '_mirror' in traj and n == 0:
                    img = img[:, ::-1]
                imgs[t, n] = cv2.resize(img, target_dim, interpolation=cv2.INTER_AREA)
        obs_dict['images'] = imgs
        if infer_gripper:
            policy_shape = policy_out[0]['actions'].shape[0]
            assert policy_shape == 4 or policy_shape == 5, "Invalid dims to infer gripper"
            if policy_shape == 4:
                for i, p in enumerate(policy_out):
                    new_action = np.ones(5, dtype = p['actions'].dtype)
                    new_action[:-1] = p['actions']
                    if obs_dict['state'][i + 1, -1] <= -0.5:
                        new_action[-1] = -1
                    p['actions'] = new_action
            elif policy_shape == 5 and seperate and 'goal_reached' not in agent_data:
                good_states = np.logical_and(obs_dict['state'][:-1, 2] >= 0.9, obs_dict['state'][:-1, -1] > -0.5)
                agent_data['goal_reached'] = np.sum(np.logical_and(np.abs(obs_dict['state'][:-1, -1]) < 0.97, good_states)) >= 2

        if seperate and not 'goal_reached' in agent_data:
            state = obs_dict['state']
            finger_sensor = obs_dict['finger_sensors']
            good_states = np.logical_and(state[:-1, 2] >= 0.9, state[:-1, -1] > 0)
            agent_data['goal_reached'] = np.sum(np.logical_and(finger_sensor[:-1, 0] > 0, good_states)) >= 2

        if 'stats' in agent_data:     #due to bug in genral_agent some robot trajs have a stats key that should be ignored
            assert agent_data['stats'] is None
            agent_data.pop('stats')
        
        if separate_views:
            obs_images = obs_dict.pop('images')
            for n in range(ncam):
                agent_data_n, obs_dict_n, policy_out_n = [copy.deepcopy(x) for x in [agent_data, obs_dict, policy_out]]
                obs_dict_n['images'] = obs_images[:, n].reshape((T, 1, target_dim[1], target_dim[0], 3))
                record_queue.put((agent_data_n, obs_dict_n, policy_out_n))
        else:
            record_queue.put((agent_data, obs_dict, policy_out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', type=str, help='target save path for record files')
    parser.add_argument('paths', type=str, help="Colon seperated list of paths to raw files")
    parser.add_argument('target_width', type=int, help='Target width to resize images')
    parser.add_argument('--split', type=float, nargs='+', default=[0.9, 0.05, 0.05])
    parser.add_argument('--T', type=int, help='agent trajectory time_sequence length', default=30)
    parser.add_argument('--offset', type=int, help='offset record counter (aka if records already exist)', default=0)
    parser.add_argument('--nworkers', type=int, help='use multiple threads or not', default=1)
    parser.add_argument('--traj_per_file', type=int, help='number of trajectories per file', default=16)
    parser.add_argument('--seperate', dest='seperate_good', action='store_true', default=False, help="seperates good and bad trajectories")
    parser.add_argument('--infer_gripper', action='store_true', default=False, help="adds gripper action to adim=4 trajs")
    parser.add_argument('--separate_views', action='store_true', default=False, help="create a separate record for each view in trajectory")

    args = parser.parse_args()
    assert sum(args.split) == 1 and not any([i < 0 or i > 1 for i in args.split]), "Split must be valid distrib"

    traj_files = []
    for path in args.paths.split(':'):
        path = os.path.expanduser(path)
        n_new_files = 0
        if 'traj_group' in path:
            new_files = glob.glob('{}/traj*'.format(path))
            n_new_files = len(new_files)
            traj_files = traj_files + new_files
        else:
            for t_group in glob.glob('{}/traj_group*'.format(path)):
                t_group_files = glob.glob('{}/traj*'.format(t_group))
                n_new_files += len(t_group_files)
                traj_files = traj_files + t_group_files
        print('for path {} got {} trajs'.format(path, n_new_files))

    random.shuffle(traj_files)
    print('Saving {} trajectories...'.format(len(traj_files)))

    m = Manager()
    record_queue = m.Queue()
    save_dir, T = args.save_dir, args.T
    seperate_good, traj_per_file = args.seperate_good, args.traj_per_file
    record_saver_proc = Process(target=record_worker, args=(
        record_queue, save_dir, T, seperate_good, traj_per_file, args.offset, tuple(args.split)))
    record_saver_proc.start()

    if args.nworkers > 1:
        confs = []
        split = len(traj_files) // args.nworkers
        for w in range(args.nworkers):
            start, end = w * split, (w + 1) * split
            if w == args.nworkers - 1:
                end = len(traj_files)
            workers_files = traj_files[start:end]

            save_conf = (workers_files, record_queue, T, args.target_width,
                         args.seperate_good, args.infer_gripper, args.separate_views)
            confs.append(save_conf)

        p = Pool(args.nworkers)
        p.map(save_worker, confs)
    else:
        save_worker((traj_files, record_queue, T, args.target_width, args.seperate_good,
                     args.infer_gripper, args.separate_views))

    record_queue.put(None)
    record_saver_proc.join()
