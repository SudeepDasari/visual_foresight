import pickle as pkl
import glob
import argparse
import random
import numpy as np
from visual_mpc.utils.im_utils import npy_to_gif
import cv2
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help='directory containing traj files (if multiple seperate with colons)')
    parser.add_argument('--T', type=int, default=30)
    parser.add_argument('--ncam', type=int, default=2)
    parser.add_argument('--nimages', type=int, default=100)
    parser.add_argument('--dim_bad', action='store_true', default=False)
    parser.add_argument('--calc_deltas', action='store_true', default=False)
    parser.add_argument('--robot_lift_rule', action='store_true', default=False)
    parser.add_argument('--im_per_row', type=int, default=5)
    args = parser.parse_args()
    assert args.nimages % args.im_per_row == 0, "--nimages should be a multiple of --im_per_row"

    traj_names = []
    for dir in args.input_dir.split(':'):
        traj_names = traj_names + glob.glob('{}/traj*'.format(dir))
    random.shuffle(traj_names)

    img_summaries = [[[] for _ in range(args.im_per_row)] for t in range(args.T)]
    summary_counter = 0
    delta_sums, rollout_fails, num_good = [], [], 0

    for t in traj_names:
        if args.calc_deltas:
            delta_sums.append(pkl.load(open('{}/obs_dict.pkl'.format(t), 'rb'))['control_delta'][1:])
        agent_data = pkl.load(open('{}/agent_data.pkl'.format(t), 'rb'))

        if args.robot_lift_rule:
            obs_dict = pkl.load(open('{}/obs_dict.pkl'.format(t), 'rb'))
            state = obs_dict['state']
            good_states = np.logical_and(state[:-1, 2] >= 0.9, state[:-1, -1] > -0.5)
            if np.sum(np.logical_and(np.abs(state[:-1, -1]) < 0.97, good_states)) >= 2:
                num_good += 1
                agent_data['goal_reached'] = True
                print('traj {} is good!'.format(t))

            else:
                agent_data['goal_reached'] = False

        elif agent_data.get('goal_reached', False):
            num_good += 1
            print('traj {} is good'.format(t))
        rollout_fails.append(agent_data.get('extra_resets', 0))

        if summary_counter < args.nimages:
            for i in range(args.T):
                frame_imgs = []
                for n in range(args.ncam):
                    fname = '{}/images{}/im_{}'.format(t, n, i)
                    if os.path.exists('{}.jpg'.format(fname)):
                        fname = fname + '.jpg'
                    elif os.path.exists('{}.png'.format(fname)):
                        fname = fname + '.png'
                    else:
                        raise ValueError

                    img_t = cv2.imread(fname)[:, :, ::-1]
                    if args.dim_bad and not agent_data['goal_reached']:
                        img_t = (img_t / 3.).astype(np.uint8)
                    frame_imgs.append(img_t)
                img_summaries[i][int(summary_counter % args.im_per_row)].append(np.concatenate(frame_imgs, axis=1))
            summary_counter += 1

    if args.calc_deltas:
        delta_sums = np.array(delta_sums)
        adim = delta_sums.shape[-1]
        print('mean deltas: {}'.format(np.sum(np.sum(delta_sums, axis=0), axis = 0) / (args.T * len(traj_names))))
        print('median delta: {}, max delta: {}'.format(np.median(delta_sums.reshape(-1, adim), axis = 0), np.amax(delta_sums.reshape(-1, adim),axis=0)))
        tmaxs = np.argmax(delta_sums[:, :, -1], axis = -1)
        traj_max = np.argmax(delta_sums[np.arange(len(traj_names)), tmaxs, -1])
        print('max degree dif at traj: {}, t: {}'.format(traj_names[traj_max], tmaxs[traj_max]))

    print(' perc good: {}, and avg num failed rollouts: {}'.format(num_good / float(len(traj_names)), np.mean(rollout_fails)))

    if args.nimages > 0:
        img_summaries = [np.concatenate([np.concatenate(row, axis=0)
                                         for row in frame_t], axis = 1) for frame_t in img_summaries]
        npy_to_gif(img_summaries, './summaries')

