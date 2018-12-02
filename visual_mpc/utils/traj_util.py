import os
import glob
import numpy as np
import re
import random


def get_maxtraj(sourcedir):
    if not os.path.exists(sourcedir):
        raise ValueError('path {} does not exist!'.format(sourcedir))

    # go to first source and get group names:
    groupnames = glob.glob(os.path.join(sourcedir, '*'))
    groupnames = [str.split(n, '/')[-1] for n in groupnames]
    gr_ind = []
    for grname in groupnames:
        try:
            gr_ind.append(int(re.match('.*?([0-9]+)$', grname).group(1)))
        except:
            continue
    max_gr = np.max(np.array(gr_ind))

    trajdir = sourcedir + "/traj_group{}".format(max_gr)
    trajname_l = glob.glob(trajdir +'/*')
    trajname_l = [str.split(n, '/')[-1] for n in trajname_l]
    traj_num = []
    for trname in trajname_l:
        traj_num.append(int(re.match('.*?([0-9]+)$', trname).group(1)))

    max_traj = np.max(np.array(traj_num))
    return max_traj


def make_traj_name_list(conf, start_end_grp = None, shuffle=True):
    combined_list = []
    for source_dir in conf['source_basedirs']:
        # assert source_dir.split('/')[-1] == 'train' or source_dir.split('/')[-1] == 'test'
        traj_per_gr = conf['ngroup']
        max_traj = get_maxtraj(source_dir)

        if start_end_grp != None:
            startgrp = start_end_grp[0]
            startidx = startgrp*traj_per_gr
            endgrp = start_end_grp[1]
            if max_traj < (endgrp+1)*traj_per_gr -1:
                endidx = max_traj
            else:
                endidx = (endgrp+1)*traj_per_gr -1
        else:
            endidx = max_traj
            startgrp = 0
            startidx = 0
            endgrp = endidx // traj_per_gr

        trajname_ind_l = []  # list of tuples (trajname, ind) where ind is 0,1,2 in range(self.split_seq_by)
        for gr in range(startgrp, endgrp + 1):  # loop over groups
            gr_dir_main = source_dir +'/traj_group' + str(gr)

            if gr == startgrp:
                trajstart = startidx
            else:
                trajstart = gr * traj_per_gr
            if gr == endgrp:
                trajend = endidx
            else:
                trajend = (gr + 1) * traj_per_gr - 1

            for i_tra in range(trajstart, trajend + 1):
                trajdir = gr_dir_main + "/traj{}".format(i_tra)
                if not os.path.exists(trajdir):
                    print('file {} not found!'.format(trajdir))
                    continue
                trajname_ind_l.append(trajdir)

        print('source_basedir: {}, length: {}'.format(source_dir,len(trajname_ind_l)))
        assert len(trajname_ind_l) == len(set(trajname_ind_l))  #check for duplicates
        combined_list += trajname_ind_l

    if shuffle:
        random.shuffle(combined_list)

    return combined_list