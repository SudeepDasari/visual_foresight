import argparse
import glob
import cv2
from visual_mpc.utils.im_utils import npy_to_gif
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Saves summary dataset gifs")
    parser.add_argument('traj_dir', type=str, help="directory containing traj groups")
    parser.add_argument('--save_dir', type=str, help="directory to save summaries", default='./')
    parser.add_argument('--per_row', type=int, help='number of tiles per row', default=10)
    parser.add_argument('--summary_freq', type=int, help="summary_freq = total_traj / # of tiles", default=100)
    parser.add_argument('--height', type=int, help="height of images", default=96)
    args = parser.parse_args()
    args.traj_dir = os.path.expanduser(args.traj_dir)

    traj_list = glob.glob('{}/*/traj*'.format(args.traj_dir))
    traj_list = sorted(traj_list, key = lambda x: int(x.split('/')[-1].split('traj')[1]))[::args.summary_freq]

    for t in traj_list:
        traj_name = t.split('/')[-1]
        n_imgs = len(glob.glob('{}/images0/*.jpg'.format(t)))
        imgs = [cv2.imread('{}/images0/im_{}.jpg'.format(t, i))[:, :, ::-1] for i in range(n_imgs)]
        
        method, old_height = cv2.INTER_AREA, imgs[0].shape[0]
        if old_height < args.height:
            method = cv2.INTER_LINEAR
        new_width = int(imgs[0].shape[1] * args.height / float(old_height))
        igms = [cv2.resize(i, (new_width, args.height), interpolation=method) for i in imgs]

        npy_to_gif(imgs, '{}/{}'.format(args.save_dir, traj_name)) 
