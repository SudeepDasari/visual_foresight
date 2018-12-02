import tensorflow as tf
import numpy as np
import os
import pickle
from .util.layers import instance_norm
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from .util.read_record import build_tfrecord_input as build_tfrecord_fn
from .util.make_plots import make_plots

from .util.online_reader import OnlineReader
import tensorflow.contrib.slim as slim
from .util.colorize_tf import colorize
from .util.losses import charbonnier_loss, ternary_loss
import collections


def length_sq(x):
    return tf.reduce_sum(tf.square(x), 3, keep_dims=True)


def length(x):
    return tf.sqrt(tf.reduce_sum(tf.square(x), 3))


def mean_square(x):
    return tf.reduce_mean(tf.square(x))


def flow_smooth_cost(flow, norm, mode, mask):
    """
    computes the norms of the derivatives and averages over the image
    :param flow_field:

    :return:
    """
    if mode == '2nd':  # compute 2nd derivative
        filter_x = [[0, 0, 0],
                    [1, -2, 1],
                    [0, 0, 0]]
        filter_y = [[0, 1, 0],
                    [0, -2, 0],
                    [0, 1, 0]]
        filter_diag1 = [[1, 0, 0],
                        [0, -2, 0],
                        [0, 0, 1]]
        filter_diag2 = [[0, 0, 1],
                        [0, -2, 0],
                        [1, 0, 0]]
        weight_array = np.ones([3, 3, 1, 4])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2

    elif mode == 'sobel':
        filter_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]  # sobel filter
        filter_y = np.transpose(filter_x)
        weight_array = np.ones([3, 3, 1, 2])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y

    weights = tf.constant(weight_array, dtype=tf.float32)

    flow_u, flow_v = tf.split(axis=3, num_or_size_splits=2, value=flow)
    delta_u =  tf.nn.conv2d(flow_u, weights, strides=[1, 1, 1, 1], padding='SAME')
    delta_v =  tf.nn.conv2d(flow_v, weights, strides=[1, 1, 1, 1], padding='SAME')

    deltas = tf.concat([delta_u, delta_v], axis=3)

    return norm(deltas, mask)


def get_coords(img_shape):
    """
    returns coordinate grid corresponding to identity appearance flow
    :param img_shape:
    :return:
    """
    y = tf.cast(tf.range(img_shape[1]), tf.float32)
    x = tf.cast(tf.range(img_shape[2]), tf.float32)
    batch_size = img_shape[0]

    X, Y = tf.meshgrid(x, y)
    coords = tf.expand_dims(tf.stack((X, Y), axis=2), axis=0)
    coords = tf.tile(coords, [batch_size, 1, 1, 1])
    return coords


def resample_layer(src_img, warp_pts, name="tgt_img"):
    with tf.variable_scope(name):
        return tf.contrib.resampler.resampler(src_img, warp_pts)


def warp_pts_layer(flow_field, name="warp_pts"):
    with tf.variable_scope(name):
        img_shape = flow_field.get_shape().as_list()
        return flow_field + get_coords(img_shape)


def apply_warp(I0, flow_field):
    warp_pts = warp_pts_layer(flow_field)
    return resample_layer(I0, warp_pts), warp_pts


class GoalDistanceNet(object):
    def __init__(self,
                 conf = None,
                 build_loss=True,
                 load_data = True,
                 I0=None,
                 I1=None,
                 iter_num = None,
                 pred_images = None,
                 load_testimages=False,
                 shuffle_buffer=512,
                 ):

        if conf['normalization'] == 'in':
            self.normalizer_fn = instance_norm
        elif conf['normalization'] == 'None':
            self.normalizer_fn = lambda x: x
        else:
            raise ValueError('Invalid layer normalization %s' % conf['normalization'])

        self.conf = conf

        self.train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")

        self.seq_len = self.conf['sequence_length']
        self.bsize = self.conf['batch_size']

        if 'row_start' in self.conf:
            self.img_height = self.conf['row_end'] - self.conf['row_start']
            self.img_width = self.conf['orig_size'][1]
        else:
            self.img_height = self.conf['orig_size'][0]
            self.img_width = self.conf['orig_size'][1]

        if load_data:
            self.iter_num = tf.placeholder(tf.float32, [], name='iternum')
            if 'new_loader' in conf:
                from python_visual_mpc.visual_mpc_core.Datasets.base_dataset import BaseVideoDataset
                train_images, val_images = [], []
                print('loading images for view {}'.format(conf['view']))
                for path, batch_size in conf['data_dir'].items():
                    data_conf = {'sequence_length': conf['sequence_length'], 'buffer_size': shuffle_buffer}
                    dataset = BaseVideoDataset(path, batch_size, data_conf)
                    train_images.append(dataset['images', 'train'][:,:, conf['view']])
                    val_images.append(dataset['images', 'val'][:,:, conf['view']])

                train_images, val_images = tf.concat(train_images, 0), tf.concat(val_images, 0)
                self.images = tf.cast(tf.cond(self.train_cond > 0,
                                 # if 1 use trainigbatch else validation batch
                                 lambda: train_images,
                                 lambda: val_images), tf.float32) / 255.0
                print('loaded images tensor: {}'.format(self.images))
            else:
                train_dict = build_tfrecord_fn(conf, mode='train')
                val_dict = build_tfrecord_fn(conf, mode='val')
                dict = tf.cond(self.train_cond > 0,
                                 # if 1 use trainigbatch else validation batch
                                 lambda: train_dict,
                                 lambda: val_dict)
                self.images = dict['images']

            if 'temp_divide_and_conquer' not in self.conf:
                self.I0, self.I1, self.I2 = self.sel_images()

        elif I0 == None:  #feed values at test time
            self.I0 = self.I0_pl= tf.placeholder(tf.float32, name='images',
                                    shape=(conf['batch_size'], self.img_height, self.img_width, 3))
            self.I1 = self.I1_pl= tf.placeholder(tf.float32, name='images',
                                     shape=(conf['batch_size'], self.img_height, self.img_width, 3))
        else:
            self.I0 = I0
            self.I1 = I1

        self.occ_fwd = tf.zeros([self.bsize,  self.img_height,  self.img_width, 1])
        self.occ_bwd = tf.zeros([self.bsize,  self.img_height,  self.img_width, 1])

        self.avg_gtruth_flow_err_sum = None
        self.build_loss = build_loss
        if build_loss:
            self.lr = tf.placeholder_with_default(self.conf['learning_rate'], ())
        self.losses = {}

    def build_net(self):
        if 'fwd_bwd' in self.conf:
            with tf.variable_scope('warpnet'):
                self.warped_I0_to_I1, self.warp_pts_bwd, self.flow_bwd, h6_bwd = self.warp(self.I0, self.I1)
            with tf.variable_scope('warpnet', reuse=True):
                self.warped_I1_to_I0, self.warp_pts_fwd, self.flow_fwd, h6_fwd = self.warp(self.I1, self.I0)

            bwd_flow_warped_fwd, _ = apply_warp(self.flow_bwd, self.flow_fwd)
            self.diff_flow_fwd = self.flow_fwd + bwd_flow_warped_fwd

            fwd_flow_warped_bwd, _ = apply_warp(self.flow_fwd, self.flow_bwd)
            self.diff_flow_bwd = self.flow_bwd + fwd_flow_warped_bwd

            if 'hard_occ_thresh' in self.conf:
                print('doing hard occ thresholding')
                mag_sq = length_sq(self.flow_fwd) + length_sq(self.flow_bwd)

                if 'occ_thres_mult' in self.conf:
                    occ_thres_mult = self.conf['occ_thres_mult']
                    occ_thres_offset = self.conf['occ_thres_offset']
                else:
                    occ_thres_mult = 0.01
                    occ_thres_offset = 0.5

                occ_thresh = occ_thres_mult * mag_sq + occ_thres_offset
                self.occ_fwd = tf.cast(length_sq(self.diff_flow_fwd) > occ_thresh, tf.float32)
                self.occ_fwd = tf.reshape(self.occ_fwd, [self.bsize,self.img_height, self.img_width,1])
                self.occ_bwd = tf.cast(length_sq(self.diff_flow_bwd) > occ_thresh, tf.float32)
                self.occ_bwd = tf.reshape(self.occ_bwd, [self.bsize, self.img_height, self.img_width, 1])
            else:
                bias = self.conf['occlusion_handling_bias']
                scale = self.conf['occlusion_handling_scale']
                diff_flow_fwd_sqlen = tf.reduce_sum(tf.square(self.diff_flow_fwd), axis=3)
                diff_flow_bwd_sqlen = tf.reduce_sum(tf.square(self.diff_flow_bwd), axis=3)
                self.occ_fwd = tf.nn.sigmoid(diff_flow_fwd_sqlen * scale + bias)  # gets 1 if occluded 0 otherwise
                self.occ_bwd = tf.nn.sigmoid(diff_flow_bwd_sqlen * scale + bias)

            self.gen_I1 = self.warped_I0_to_I1
            self.gen_I0 = self.warped_I1_to_I0
        else:
            self.warped_I0_to_I1, self.warp_pts_bwd, self.flow_bwd, _ = self.warp(self.I0, self.I1)

            self.gen_I1 = self.warped_I0_to_I1
            self.gen_I0, self.flow_fwd = None, None

        self.occ_mask_bwd = 1 - self.occ_bwd  # 0 at occlusion
        self.occ_mask_fwd = 1 - self.occ_fwd

        if self.build_loss:
            # image_summary:
            if 'fwd_bwd' in self.conf:
                self.add_pair_loss(I1=self.I1, gen_I1=self.gen_I1, occ_bwd=self.occ_bwd, flow_bwd=self.flow_bwd, diff_flow_bwd=self.diff_flow_bwd,
                                   I0=self.I0, gen_I0=self.gen_I0, occ_fwd=self.occ_fwd, flow_fwd=self.flow_fwd, diff_flow_fwd=self.diff_flow_fwd)

                self.image_summaries = self.build_image_summary(
                    [self.I0, self.I1, self.gen_I0, self.gen_I1, length(self.flow_bwd), length(self.flow_fwd), self.occ_mask_bwd, self.occ_mask_fwd])
            else:
                self.add_pair_loss(I1=self.I1, gen_I1=self.gen_I1,  flow_bwd=self.flow_bwd)
                self.image_summaries = self.build_image_summary([self.I0, self.I1, self.gen_I1, length(self.flow_bwd)])

            self.combine_losses()

    def sel_images(self):
        max_deltat = self.conf['max_deltat']
        t_fullrange = 2e4
        delta_t = tf.cast(tf.ceil(max_deltat * (tf.cast(self.iter_num + 1, tf.float32)) / t_fullrange), dtype=tf.int32)
        delta_t = tf.clip_by_value(delta_t, 1, max_deltat-1)
        self.delta_t = delta_t

        self.tstart = tf.random_uniform([1], 0, self.conf['sequence_length'] - delta_t, dtype=tf.int32)

        if 'deterministic_increase_tdist' in self.conf:
            self.tend = self.tstart + delta_t
        else:
            minval = tf.ones([], dtype=tf.int32)
            self.tend = self.tstart + tf.random_uniform([1], minval, delta_t + 1, dtype=tf.int32)

        begin = tf.stack([0, tf.squeeze(self.tstart), 0, 0, 0],0)
        I0 = tf.squeeze(tf.slice(self.images, begin, [-1, 1, -1, -1, -1]))

        begin = tf.stack([0, tf.squeeze(self.tend), 0, 0, 0], 0)
        I1 = tf.squeeze(tf.slice(self.images, begin, [-1, 1, -1, -1, -1]))

        return I0, I1, None

    def conv_relu_block(self, input, out_ch, k=3, upsmp=False, n_layer=1):
        h = input
        for l in range(n_layer):
            h = slim.layers.conv2d(  # 32x32x64
                h,
                out_ch, [k, k],
                stride=1)
            h = self.normalizer_fn(h)

        if upsmp:
            mult = 2
        else: mult = 0.5
        imsize = np.array(h.get_shape().as_list()[1:3])*mult

        h = tf.image.resize_images(h, imsize, method=tf.image.ResizeMethod.BILINEAR)
        return h

    def warp(self, source_image, dest_image):
        """
        warps I0 onto I1
        :param source_image:
        :param dest_image:
        :return:
        """

        if 'ch_mult' in self.conf:
            ch_mult = self.conf['ch_mult']
        else: ch_mult = 1

        if 'late_fusion' in self.conf:
            print('building late fusion net')
            with tf.variable_scope('pre_proc_source'):
                h3_1 = self.pre_proc_net(source_image, ch_mult)
            with tf.variable_scope('pre_proc_dest'):
                h3_2 = self.pre_proc_net(dest_image, ch_mult)
            h3 = tf.concat([h3_1, h3_2], axis=3)
        else:
            I0_I1 = tf.concat([source_image, dest_image], axis=3)
            with tf.variable_scope('h1'):
                h1 = self.conv_relu_block(I0_I1, out_ch=32*ch_mult)  #24x32x3

            with tf.variable_scope('h2'):
                h2 = self.conv_relu_block(h1, out_ch=64*ch_mult)  #12x16x3

            with tf.variable_scope('h3'):
                h3 = self.conv_relu_block(h2, out_ch=128*ch_mult)  #6x8x3

            if self.conf['orig_size'][1] == 128:
                with tf.variable_scope('h3_1'):
                    h3 = self.conv_relu_block(h3, out_ch=256*ch_mult)  #6x8x3
                with tf.variable_scope('h3_2'):
                    h3 = self.conv_relu_block(h3, out_ch=64*ch_mult, upsmp=True)  #12x16x3

        with tf.variable_scope('h4'):
            h4 = self.conv_relu_block(h3, out_ch=64*ch_mult, upsmp=True)  #12x16x3

        with tf.variable_scope('h5'):
            h5 = self.conv_relu_block(h4, out_ch=32*ch_mult, upsmp=True)  #24x32x3

        with tf.variable_scope('h6'):
            h6 = self.conv_relu_block(h5, out_ch=16*ch_mult, upsmp=True)  #48x64x3

        with tf.variable_scope('h7'):
            flow_field = slim.layers.conv2d(
                h6,  2, [5, 5], stride=1, activation_fn=None)

        warp_pts = warp_pts_layer(flow_field)
        gen_image = resample_layer(source_image, warp_pts)

        return gen_image, warp_pts, flow_field, h6

    def pre_proc_net(self, input, ch_mult):
        with tf.variable_scope('h1'):
            h1 = self.conv_relu_block(input, out_ch=32 * ch_mult)  # 24x32x3
        with tf.variable_scope('h2'):
            h2 = self.conv_relu_block(h1, out_ch=64 * ch_mult)  # 12x16x3
        with tf.variable_scope('h3'):
            h3 = self.conv_relu_block(h2, out_ch=128 * ch_mult)  # 6x8x3
        return h3

    def build_image_summary(self, tensors, numex=16, name=None, suf=''):
        """
        takes numex examples from every tensor and concatentes examples side by side
        and the different tensors from top to bottom
        :param tensors:
        :param numex:
        :return:
        """
        ten_list = []
        for ten in tensors:
            if len(ten.get_shape().as_list()) == 3 or ten.get_shape().as_list()[-1] == 1:
                ten_colored = []
                for b in range(ten.get_shape().as_list()[0]):
                    ten_colored.append(colorize(ten[b], tf.reduce_min(ten), tf.reduce_max(ten), 'viridis'))
                ten = tf.stack(ten_colored, axis=0)

            unstacked = tf.unstack(ten, axis=0)[:numex]
            concated = tf.concat(unstacked, axis=1)
            ten_list.append(concated)
        combined = tf.concat(ten_list, axis=0)
        combined = tf.reshape(combined, [1]+combined.get_shape().as_list())

        if name ==None:
            name = 'Images' + suf
        return tf.summary.image(name, combined)

    def add_pair_loss(self, I1, gen_I1, occ_bwd=None, flow_bwd=None, diff_flow_fwd=None,
                      I0=None, gen_I0=None, occ_fwd=None, flow_fwd=None, diff_flow_bwd=None, mult=1., suf=''):
        if occ_bwd is not None:
            occ_mask_bwd = 1 - occ_bwd  # 0 at occlusion
        else:
            occ_mask_bwd = tf.ones(I1.get_shape().as_list()[:3] + [1])

        if occ_fwd is not None:
            occ_mask_fwd = 1 - occ_fwd

        if self.conf['norm'] == 'l2':
            norm = mean_square
        elif self.conf['norm'] == 'charbonnier':
            norm = charbonnier_loss
        else: raise ValueError("norm not defined!")

        newlosses = {}
        if 'ternary_loss' in self.conf:
            newlosses['ternary_I1'+suf] = ternary_loss(I1, gen_I1, occ_mask_bwd)*self.conf['ternary_loss']
        newlosses['train_I1_recon_cost'+suf] = norm((gen_I1 - I1), occ_mask_bwd)

        if 'fwd_bwd' in self.conf:
            if 'ternary_loss' in self.conf:
                newlosses['ternary_I0'+suf] = ternary_loss(I0, gen_I0, occ_mask_fwd)*self.conf['ternary_loss']
            newlosses['train_I0_recon_cost'+suf] = norm((gen_I0 - I0), occ_mask_fwd)

            fd = self.conf['flow_diff_cost']
            newlosses['train_flow_diff_cost'+suf] = (norm(diff_flow_fwd, occ_mask_fwd)
                                                     +norm(diff_flow_bwd, occ_mask_bwd)) * fd

            if 'occlusion_handling' in self.conf:
                occ = self.conf['occlusion_handling']
                newlosses['train_occlusion_handling'+suf] = (tf.reduce_mean(occ_fwd) + tf.reduce_mean(occ_bwd)) * occ

        if 'smoothcost' in self.conf:
            sc = self.conf['smoothcost']
            newlosses['train_smooth_bwd'+suf] = flow_smooth_cost(flow_bwd, norm, self.conf['smoothmode'],
                                                                    occ_mask_bwd) * sc
            if 'fwd_bwd' in self.conf:
                newlosses['train_smooth_fwd'+suf] = flow_smooth_cost(flow_fwd, norm, self.conf['smoothmode'],
                                                            occ_mask_fwd) * sc
        if 'flow_penal' in self.conf:
            if 'fwd_bwd' in self.conf:
                newlosses['flow_penal'+suf] = (tf.reduce_mean(tf.square(flow_bwd)) +
                                                tf.reduce_mean(tf.square(flow_fwd))) * self.conf['flow_penal']
            else:
                newlosses['flow_penal' + suf] = (tf.reduce_mean(tf.square(flow_bwd))) * self.conf['flow_penal']

        for k in list(newlosses.keys()):
            self.losses[k] = newlosses[k]*mult

    def combine_losses(self):
        train_summaries = []
        val_summaries = []
        self.loss = 0
        for k in list(self.losses.keys()):
            single_loss = self.losses[k]
            self.loss += single_loss
            train_summaries.append(tf.summary.scalar(k, single_loss))
        train_summaries.append(tf.summary.scalar('train_total', self.loss))
        train_summaries.append(tf.summary.scalar('delta_t', self.delta_t))
        val_summaries.append(tf.summary.scalar('val_total', self.loss))
        self.train_summ_op = tf.summary.merge(train_summaries)
        self.val_summ_op = tf.summary.merge(val_summaries)

        self.global_step = tf.Variable(0, name='global_step',trainable=False)
        if 'decay_lr' in self.conf:
            self.learning_rate = tf.train.exponential_decay(self.lr, self.global_step,
                                                       4000, 0.96, staircase=True)
            print('using exponentially decayed lr')
        else:
            self.learning_rate = tf.constant(self.conf['learning_rate'])
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, self.global_step)


    def visualize(self, sess):
        if 'compare_gtruth_flow' in self.conf:
            flow_errs, flow_errs_mean, flow_mags_bwd, flow_mags_bwd_gtruth, gen_images_I1, images = self.compute_bench(self, sess)
            with open(self.conf['output_dir'] + '/gtruth_flow_err.txt', 'w') as f:
                flow_errs_flat = np.stack(flow_errs_mean).flatten()
                string = 'average one-step flowerrs on {} example trajectories mean {} std err of the mean {} \n'.format(self.bsize,
                                np.mean(flow_errs_flat), np.std(flow_errs_flat)/np.sqrt(flow_errs_flat.shape[0]))
                print(string)
                f.write(string)
            print('written output to ',self.conf['output_dir'] + '/gtruth_flow_err.txt')

            videos = collections.OrderedDict()
            videos['I0_ts'] = np.split(images, images.shape[1], axis=1)
            videos['gen_images_I1'] = [np.zeros_like(gen_images_I1[0])] +gen_images_I1
            videos['flow_mags_bwd'] = [np.zeros_like(flow_mags_bwd[0])] + flow_mags_bwd
            videos['flow_mags_bwd_gtruth'] = [np.zeros_like(flow_mags_bwd_gtruth[0])]+ flow_mags_bwd_gtruth
            videos['flow_errs'] = ([np.zeros_like(np.zeros_like(flow_errs[0]))] + flow_errs,
                                   [np.zeros_like(flow_errs_mean[0])] + flow_errs_mean)
            # videos['gen_image_I1_gtruthwarp_l'] = [np.zeros_like(gen_image_I1_gtruthwarp_l[0])] + gen_image_I1_gtruthwarp_l

            num_ex = 4
            for k in list(videos.keys()):
                if isinstance(videos[k], tuple):
                    videos[k] = ([el[:num_ex] for el in videos[k][0]], [el[:num_ex] for el in videos[k][1]])
                else:
                    videos[k] = [el[:num_ex] for el in videos[k]]

            name = str.split(self.conf['output_dir'], '/')[-2]
            dict = {'videos': videos, 'name': 'flow_err_' + name}

            pickle.dump(dict, open(self.conf['output_dir'] + '/data.pkl', 'wb'))
            make_plots(self.conf, dict=dict)
            return

        else:  # when visualizing sequence of warps from video
            videos = build_tfrecord_fn(self.conf, mode='test')
            if 'vidpred_data' in self.conf:
                images, pred_images = sess.run([videos['images'], videos['gen_images']])
                pred_images = np.squeeze(pred_images)
            else:
                [images] = sess.run([videos['images']])

        num_examples = self.conf['batch_size']
        I1 = images[:, -1]

        gen_images_I0 = []
        gen_images_I1 = []
        I0_t_reals = []
        I0_ts = []
        flow_mags_bwd = []
        flow_mags_fwd = []
        occ_bwd_l = []
        occ_fwd_l = []
        warpscores_bwd = []
        warpscores_fwd = []

        bwd_flows_l = []
        fwd_flows_l = []
        warp_pts_bwd_l= []
        warp_pts_fwd_l= []

        for t in range(self.conf['sequence_length']-1):
            if 'vidpred_data' in self.conf:
                I0_t = pred_images[:, t]
                I0_t_real = images[:, t]

                I0_t_reals.append(I0_t_real)
            else:
                I0_t = images[:, t]

            I0_ts.append(I0_t)

            if 'fwd_bwd' in self.conf:
                [gen_image_I1, bwd_flow, occ_bwd, norm_occ_mask_bwd,
                 gen_image_I0, fwd_flow, occ_fwd, norm_occ_mask_fwd, warp_pts_bwd, warp_pts_fwd] = sess.run([self.gen_I1,
                                                                                 self.flow_bwd,
                                                                                 self.occ_bwd,
                                                                                 self.occ_mask_bwd,
                                                                                 self.gen_I0,
                                                                                 self.flow_fwd,
                                                                                 self.occ_fwd,
                                                                                 self.occ_mask_fwd,
                                                                                 self.warp_pts_bwd,
                                                                                 self.warp_pts_fwd,
                                                                                 ], {self.I0_pl: I0_t, self.I1_pl: I1})
                occ_bwd_l.append(occ_bwd)
                occ_fwd_l.append(occ_fwd)

                gen_images_I0.append(gen_image_I0)

                fwd_flows_l.append(fwd_flow)
                warp_pts_fwd_l.append(warp_pts_fwd)
            else:
                [gen_image_I1, bwd_flow] = sess.run([self.gen_I1, self.flow_bwd], {self.I0_pl:I0_t, self.I1_pl: I1})

            bwd_flows_l.append(bwd_flow)
            warp_pts_bwd_l.append(warp_pts_bwd)
            gen_images_I1.append(gen_image_I1)

            flow_mag_bwd = np.linalg.norm(bwd_flow, axis=3)
            flow_mags_bwd.append(flow_mag_bwd)
            if 'fwd_bwd' in self.conf:
                flow_mag_fwd = np.linalg.norm(fwd_flow, axis=3)
                flow_mags_fwd.append(flow_mag_fwd)
                warpscores_bwd.append(np.mean(np.mean(flow_mag_bwd * np.squeeze(norm_occ_mask_bwd), axis=1), axis=1))
                warpscores_fwd.append(np.mean(np.mean(flow_mag_fwd * np.squeeze(norm_occ_mask_fwd), axis=1), axis=1))
            else:
                warpscores_bwd.append(np.mean(np.mean(flow_mag_bwd, axis=1), axis=1))

            # flow_mags.append(self.color_code(flow_mag, num_examples))

        videos = collections.OrderedDict()
        videos['I0_ts'] = I0_ts
        videos['gen_images_I1'] = gen_images_I1
        videos['flow_mags_bwd'] = (flow_mags_bwd, warpscores_bwd)
        videos['bwd_flow'] = bwd_flows_l
        videos['warp_pts_bwd'] = warp_pts_bwd_l

        if 'vidpred_data' in self.conf:
            videos['I0_t_real'] = I0_t_reals

        if 'fwd_bwd' in self.conf:
            videos['warp_pts_fwd'] = warp_pts_fwd_l
            videos['fwd_flow'] = fwd_flows_l
            videos['occ_bwd'] = occ_bwd_l
            videos['gen_images_I0'] = gen_images_I0
            videos['flow_mags_fwd'] = (flow_mags_fwd, warpscores_fwd)
            videos['occ_fwd'] = occ_fwd_l

        name = str.split(self.conf['output_dir'], '/')[-2]
        dict = {'videos':videos, 'name':name, 'I1':I1}

        pickle.dump(dict, open(self.conf['output_dir'] + '/data.pkl', 'wb'))
        # make_plots(self.conf, dict=dict)

    def run_bench(self, benchmodel, sess):
        _, flow_errs_mean, _, _, _, _ = self.compute_bench(benchmodel, sess)

        flow_errs_mean = np.mean(np.stack(flow_errs_mean).flatten())
        print('benchmark result: ', flow_errs_mean)

        if self.avg_gtruth_flow_err_sum == None:
            self.flow_errs_mean_pl = tf.placeholder(tf.float32, name='flow_errs_mean_pl', shape=[])
            self.avg_gtruth_flow_err_sum = tf.summary.scalar('avg_gtruth_flow_err', self.flow_errs_mean_pl)

        return sess.run([self.avg_gtruth_flow_err_sum], {self.flow_errs_mean_pl: flow_errs_mean})[0]

    def compute_bench(self, model, sess):
        # self.conf['source_basedirs'] = [os.environ['VMPC_DATA_DIR'] + '/cartgripper_gtruth_flow/train']
        self.conf['source_basedirs'] = [os.environ['VMPC_DATA_DIR'] + '/cartgripper_gtruth_flow_masks/train']
        self.conf['sequence_length'] = 9
        tag_images = {'name': 'images',
                      'file': '/images/im{}.png',  # only tindex
                      'shape': [48, 64, 3]}
        tag_bwd_flow = {'name': 'bwd_flow',
                        'not_per_timestep': '',
                        'shape': [self.conf['sequence_length']-1, 48, 64, 2]}
        ob_masks = {'name': 'ob_masks',
                    'not_per_timestep': '',
                    'shape': [self.conf['sequence_length'], 1, 48, 64]}
        self.conf['sourcetags'] = [tag_images, tag_bwd_flow, ob_masks]
        self.conf['ngroup'] = 1000
        r = OnlineReader(self.conf, 'val', sess=sess)
        images, tag_bwd_flow, ob_masks  = r.get_batch_tensors()
        [images, gtruth_bwd_flows, ob_masks] = sess.run([images, tag_bwd_flow, tf.squeeze(ob_masks)])
        gtruth_bwd_flows = np.flip(gtruth_bwd_flows, axis=-1)  # important ! need to flip flow to make compatible
        gen_images_I1 = []
        bwd_flows = []
        flow_errs_mean = []
        flow_errs = []
        flow_mags_bwd = []
        flow_mags_bwd_gtruth = []
        for t in range(images.shape[1] - 1):
            [gen_image_I1, bwd_flow] = sess.run([model.gen_I1, model.flow_bwd], {model.I0_pl: images[:, t],
                                                                                 model.I1_pl: images[:, t + 1]})
            gen_images_I1.append(gen_image_I1)
            bwd_flows.append(bwd_flow)

            flow_diffs = bwd_flow - gtruth_bwd_flows[:, t]
            flow_errs.append(np.linalg.norm(ob_masks[:, t,:,:,None]*flow_diffs, axis=-1))
            flow_errs_mean.append(np.mean(np.mean(flow_errs[-1], axis=1), axis=1))
            flow_mags_bwd.append(np.linalg.norm(bwd_flow, axis=-1))
            flow_mags_bwd_gtruth.append(np.linalg.norm(gtruth_bwd_flows[:, t], axis=-1))
            # verify gtruth optical flow:
            # gen_image_I1_gtruthwarp = apply_warp(self.I0_pl, gtruth_bwd_flows_pl)
            # gen_image_I1_gtruthwarp_l += sess.run([gen_image_I1_gtruthwarp], {self.I0_pl: images[:, t],
            #                                                                   gtruth_bwd_flows_pl: gtruth_bwd_flows[:,t]})
        return flow_errs, flow_errs_mean, flow_mags_bwd, flow_mags_bwd_gtruth, gen_images_I1, images

    def color_code(self, input, num_examples):
        cmap = plt.cm.get_cmap()

        l = []
        for b in range(num_examples):
            f = input[b] / (np.max(input[b]) + 1e-6)
            f = cmap(f)[:, :, :3]
            l.append(f)

        return np.stack(l, axis=0)
