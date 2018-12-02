import random
import tensorflow as tf
import os
import glob
import numpy as np
import re
from collections import OrderedDict
import pickle
import threading
import itertools
import cv2
import tarfile
import time


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


def get_start_end(conf):
    """
    get start and end time indices, which will be read from trajectory
    :param conf:
    :return:
    """
    if 'total_num_img' in conf:
        end = conf['total_num_img']
        t_ev_nstep = conf['take_ev_nth_step']
    else:
        t_ev_nstep = 1
        end = conf['sequence_length']

    smp_range = end // t_ev_nstep - conf['sequence_length']
    if 'shift_window' in conf:
        print('performing shifting in time')
        start = np.random.random_integers(0, smp_range) * t_ev_nstep
    else:
        start = 0
    end = start + conf['sequence_length'] * t_ev_nstep

    if 'take_ev_nth_step' in conf:
        take_ev_nth_step = conf['take_ev_nth_step']
    else: take_ev_nth_step = 1

    return start, end, take_ev_nth_step


def read_single_img(dataind, tag_dict, tar, trajname, icam=None):
    if tar != None:
        im_filename = 'traj/images/im{}.png'.format(dataind)
        img_stream = tar.extractfile(im_filename)
        file_bytes = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
    else:
        imfile = trajname + tag_dict['file'].format(dataind)
        if not os.path.exists(imfile):
            raise ValueError("file {} does not exist!".format(imfile))
        img = cv2.imread(imfile)
    imheight = tag_dict['shape'][0]  # get target im_sizes
    imwidth = tag_dict['shape'][1]
    if 'rowstart' in tag_dict:  # get target cropping if specified
        rowstart = tag_dict['rowstart']
        colstart = tag_dict['colstart']
    # setting used in wrist_rot
    if 'shrink_before_crop' in tag_dict:
        shrink_factor = tag_dict['shrink_before_crop']
        img = cv2.resize(img, (0, 0), fx=shrink_factor, fy=shrink_factor, interpolation=cv2.INTER_AREA)
        img = img[rowstart:rowstart + imheight, colstart:colstart + imwidth]
    # setting used in softmotion30_v1
    elif 'crop_before_shrink' in tag_dict:
        raw_image_height = img.shape[0]
        img = img[rowstart:rowstart + raw_image_height, colstart:colstart + raw_image_height]
        target_res = tag_dict['target_res']
        img = cv2.resize(img, target_res, interpolation=cv2.INTER_AREA)
    elif 'rowstart' in tag_dict:
        img = img[rowstart:rowstart + imheight, colstart:colstart + imwidth]

    img = img[:, :, ::-1]  # bgr => rgb
    img = img.astype(np.float32) / 255.
    return img


def read_img(tag_dict, dataind, tar=None, trajname=None):
    """
    read a single image, either from tar-file or directly
    :param tar:  far file handle
    :param tag_dict: dictionary describing the tag
    :param dataind: the timestep in the data folder (may not be equal to the timestep used for the allocated array)
    :return:
    """
    if 'ncam' in tag_dict:
        images = []
        for icam in range(tag_dict['ncam']):
            images.append(read_single_img(dataind, tag_dict, tar, trajname, icam))
        img = np.stack(images, 0)
    else:
        img = read_single_img(dataind, tag_dict, tar, trajname)
    return img


def read_trajectory(conf, trajname, use_tar = False):
    """
    the configuration file needs:
    source_basedirs key: a list of directories where to load the data from, data is concatenated (advantage: no renumbering needed when using multiple sources)
    sourcetags: a list of tags where each tag is a dict with
        # name: the name of the data field
        # shape: the target shape, will be cropped to match this shape
        # rowstart: starting row for cropping
        # rowend: end row for cropping
        # colstart: start column for cropping
        # shrink_before_crop: shrink image according to this ratio before cropping
        # brightness_threshold: if average pixel value lower discard video
        # not_per_timestep: if this key is there, load the data for this tag at once from pkl file

    :param conf:
    :param trajname: folder of trajectory to be loaded
    :param use_tar: whether to load from tar files
    """
    t0 = time.time()
    # try:
    traj_index = int(re.match('.*?([0-9]+)$', trajname).group(1))
    nump_array_dict = {}

    if use_tar:
        tar = tarfile.open(trajname + "/traj.tar")
        pkl_file_stream = tar.extractfile('traj/agent_data.pkl')
        pkldata = pickle.load(pkl_file_stream)
    else:
        tar = None
        pkldata = pickle.load(open(trajname + '/agent_data.pkl', 'rb'), encoding='latin1')

    for tag_dict in conf['sourcetags']:
        if 'not_per_timestep' not in tag_dict:
            numpy_arr = np.zeros([conf['sequence_length']] + tag_dict['shape'], dtype=np.float32)
            nump_array_dict[tag_dict['name']] = numpy_arr

    start, end, take_ev_nth_step = get_start_end(conf)

    # remove not_per_timestep tags
    filtered_source_tags = []
    for tag_dict in conf['sourcetags']:
        if 'not_per_timestep' in tag_dict:
            nump_array_dict[tag_dict['name']] = pkldata[tag_dict['name']]
        else:
            filtered_source_tags.append(tag_dict)

    trajind = 0
    for dataind in range(start, end, take_ev_nth_step):

        for tag_dict in filtered_source_tags:
            tag_name = tag_dict['name']

            if '.pkl' in tag_dict['file']:  # if it's data from Pickle file
                if 'pkl_names' in tag_dict:  # if a tag, e.g. the the state is split up into multiple tags
                    pklread0 = pkldata[tag_dict['pkl_names'][0]]
                    pklread1 = pkldata[tag_dict['pkl_names'][1]]
                    nump_array_dict[tag_name][trajind] = np.concatenate([pklread0[dataind], pklread1[dataind]],
                                                                        axis=0)
                else:
                    nump_array_dict[tag_name][trajind] = pkldata[tag_dict['name']][dataind]
            else:  # if it's image data
                nump_array_dict[tag_name][trajind] = read_img(tag_dict, dataind, trajname=trajname, tar=tar)
        trajind += 1

    if use_tar:
        tar.close() # important: close file
    return nump_array_dict


def reading_thread(conf, subset_traj, enqueue_op, sess, placeholders, use_tar):
    num_errors = 0
    print('started process with PID:', os.getpid())

    for trajname in itertools.cycle(subset_traj):  # loop of traj0, traj1,..
        nump_array_dict = read_trajectory(conf, trajname, use_tar=use_tar)

        # print 'reading ',trajname

        feed_dict = {}
        for tag_dict in conf['sourcetags']:
            tag_name = tag_dict['name']
            feed_dict[placeholders[tag_name]] = nump_array_dict[tag_name]

        t1 = time.time()
        sess.run(enqueue_op, feed_dict=feed_dict)

        # if traj_index % 10 == 0:
        #     print 't ful enqueu', time.time() - t0
        #     print 't enqueu run', time.time() - t1

        # except KeyboardInterrupt:
        #     sys.exit()
        # except:
        #     print "error occured"
        #     num_errors += 1


class OnlineReader(object):
    def __init__(self, conf, mode, sess, use_tar=False):
        """

        :param conf:
        :param mode:  'train': shuffle data or 'test': don't shuffle
        :param sess:
        """

        self.sess = sess
        self.conf = conf
        self.mode = mode

        self.use_tar = use_tar

        self.place_holders = OrderedDict()

        pl_shapes = []
        self.tag_names = []
        # loop through tags
        for tag_dict in conf['sourcetags']:
            if 'not_per_timestep' in tag_dict:
                pl_shapes.append(tag_dict['shape'])
            else:
                pl_shapes.append([conf['sequence_length']] + tag_dict['shape'])
            self.tag_names.append(tag_dict['name'])
            self.place_holders[tag_dict['name']] = tf.placeholder(tf.float32, name=tag_dict['name'], shape=pl_shapes[-1])
        if mode == 'train' or mode == 'val':
            self.num_threads = 10
        else: self.num_threads = 1

        if mode == 'test':
            self.shuffle = False
        else: self.shuffle = True

        tf_dtypes = [tf.float32]*len(pl_shapes)

        self.q = tf.FIFOQueue(1000, tf_dtypes, shapes=pl_shapes)
        self.enqueue_op = self.q.enqueue(list(self.place_holders.values()))

        auto_split = False  # automatically divide dataset into train, val, test and save the split to pkl-file
        if auto_split:
            data_sets = self.search_data()
            self.traj_list = self.combine_traj_lists(data_sets)
        else:
            self.traj_list = make_traj_name_list(conf, shuffle=self.shuffle)

        self.start_threads(self.traj_list)

    def get_batch_tensors(self):
        tensor_list = self.q.dequeue_many(self.conf['batch_size'])
        return tensor_list


    def search_data(self):
        """
        automatically divide dataset into train, val, test and save the split to pkl-file;
        if pkl-file already exists load the split
        :return: train, val, test datasets for every source
        """

        print('searching data')
        datasets = []
        for dir in self.conf['source_basedirs']:
            source_name = str.split(dir, '/')[-1]

            print('preparing source_basedir', dir)
            split_file = self.conf['current_dir'] + '/' + source_name + '_split.pkl'

            dataset_i = {}
            if os.path.isfile(split_file):
                print('loading datasplit from ', split_file)
                dataset_i = pickle.load(open(split_file, "rb"))
            else:
                traj_list = make_traj_name_list(self.conf, shuffle=True)

                #make train, val, test split
                test_traj = traj_list[:256]  # use first 256 for test
                traj_list = traj_list[256:]
                num_traj = len(traj_list)

                index = int(np.floor(self.conf['train_val_split'] * num_traj))

                train_traj = traj_list[:index]
                val_traj = traj_list[index:]

                dataset_i['source_basedir'] = dir
                dataset_i['train'] = train_traj
                dataset_i['val'] = val_traj
                dataset_i['test'] = test_traj

                pickle.dump(dataset_i, open(split_file, 'wb'))

            datasets.append(dataset_i)

        return datasets

    def combine_traj_lists(self, datasets):
        combined = []
        for dset in datasets:
            # select whether to use train, val or test data
            dset = dset[self.mode]
            combined += dset
        if self.shuffle:
            random.shuffle(combined)
        return combined

    def start_threads(self, traj_list):
        """
            :param sourcedirs:
            :param tf_rec_dir:
            :param gif_dir:
            :param traj_name_list:
            :param crop_from_highres:
            :param start_end_grp: list with [startgrp, endgrp]
            :return:
            """
        # ray.init()

        itraj_start = 0
        n_traj = len(traj_list)

        traj_per_worker = int(n_traj / np.float32(self.num_threads))
        start_idx = [itraj_start + traj_per_worker * i for i in range(self.num_threads)]
        end_idx = [itraj_start + traj_per_worker * (i + 1) - 1 for i in range(self.num_threads)]

        for i in range(self.num_threads):
            print('worker {} going from {} to {} '.format(i, start_idx[i], end_idx[i]))
            subset_traj = traj_list[start_idx[i]:end_idx[i]]

            t = threading.Thread(target=reading_thread, args=(self.conf, subset_traj,
                                                              self.enqueue_op, self.sess,
                                                              self.place_holders, self.use_tar
                                                                ))
            t.setDaemon(True)
            t.start()