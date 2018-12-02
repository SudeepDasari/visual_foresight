from tensorflow.python.platform import gfile
import tensorflow as tf
import copy
from random import shuffle as shuffle_list
import os
import numpy as np


COLOR_CHAN = 3
def decode_im(conf, features, image_name):

    if 'orig_size' in conf:
        ORIGINAL_HEIGHT = conf['orig_size'][0]
        ORIGINAL_WIDTH = conf['orig_size'][1]
    else:
        ORIGINAL_WIDTH = 64
        ORIGINAL_HEIGHT = 64
    if 'row_start' in conf:
        IMG_HEIGHT = conf['row_end'] - conf['row_start']
    else:
        IMG_HEIGHT = ORIGINAL_HEIGHT
    if 'img_width' in conf:
        IMG_WIDTH = conf['img_width']
    else:
        IMG_WIDTH = ORIGINAL_WIDTH
    image = tf.decode_raw(features[image_name], tf.uint8)
    image = tf.reshape(image, shape=[1, ORIGINAL_HEIGHT * ORIGINAL_WIDTH * COLOR_CHAN])
    image = tf.reshape(image, shape=[ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
    if 'row_start' in conf:
        image = image[conf['row_start']:conf['row_end']]
    image = tf.reshape(image, [1, IMG_HEIGHT, IMG_WIDTH, COLOR_CHAN])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def mix_datasets(datasets, sizes):
    """Sample batch with specified mix of ground truth and generated data_files points.

    Args:
      ground_truth_x: tensor of ground-truth data_files points.
      generated_x: tensor of generated data_files points.
      batch_size: batch size
      ratio_01: ratio between examples taken from the first dataset and the batchsize
    Returns:
      New batch with num_ground_truth sampled from ground_truth_x and the rest
      from generated_x.
    """
    assert isinstance(sizes[0], int)
    batch_size = datasets[0]['images'].get_shape().as_list()[0]
    assert np.sum(np.array(sizes)) == batch_size

    output = {}
    for key in datasets[0].keys():
        sel_ten = []
        for i, d in enumerate(datasets):
            sel_ten.append(d[key][:sizes[i]])

        ten = tf.concat(sel_ten, axis=0)
        output[key] = ten
    return output


def build_tfrecord_input(conf, mode='train', input_files=None, shuffle=True, buffersize=512):
    if isinstance(conf['data_dir'], dict) and input_files==None:
        data_set = []
        ratios = []

        for key in conf['data_dir'].keys():
            conf_ = copy.deepcopy(conf)
            conf_['data_dir'] = key
            print('loading', key)
            data_set.append(build_tfrecord_single(conf_, mode, None, shuffle, buffersize))
            ratios.append(conf['data_dir'][key])
        comb_dataset = mix_datasets(data_set, ratios)

        return comb_dataset
    else:
        return build_tfrecord_single(conf, mode, input_files, shuffle, buffersize)


def build_tfrecord_single(conf, mode='train', input_files=None, shuffle=True, buffersize=512):
    """Create input tfrecord tensors.

    Args:
      training: training or validation data_files.
      conf: A dictionary containing the configuration for the experiment
    Returns:
      list of tensors corresponding to images, actions, and states. The images
      tensor is 5D, batch x time x height x width x channels. The state and
      action tensors are 3D, batch x time x dimension.
    Raises:
      RuntimeError: if no files found.
    """
    if 'sdim' in conf:
        sdim = conf['sdim']
    else: sdim = 3
    if 'adim' in conf:
        adim = conf['adim']
    else: adim = 4
    print('adim', adim)
    print('sdim', sdim)

    if input_files is not None:
        if not isinstance(input_files, list):
            filenames = [input_files]
        else: filenames = input_files
    else:
        filenames = gfile.Glob(os.path.join(conf['data_dir'], mode) + '/*')
        if mode == 'val' or mode == 'test':
            shuffle = False
        else:
            shuffle = True
        if not filenames:
            raise RuntimeError('No data_files files found.')

    print('using shuffle: ', shuffle)
    if shuffle:
        shuffle_list(filenames)
    # Reads an image from a file, decodes it into a dense tensor, and resizes it
    # to a fixed shape.
    def _parse_function(serialized_example):
        image_seq, image_main_seq, endeffector_pos_seq, gen_images_seq, gen_states_seq,\
        action_seq, object_pos_seq, robot_pos_seq, goal_image = [], [], [], [], [], [], [], [], []

        load_indx = list(range(0, conf['sequence_length'], conf['skip_frame']))
        print('using frame sequence: ', load_indx)

        rand_h = tf.random_uniform([1], minval=-0.2, maxval=0.2)
        rand_s = tf.random_uniform([1], minval=-0.2, maxval=0.2)
        rand_v = tf.random_uniform([1], minval=-0.2, maxval=0.2)
        features_name = {}

        for i in load_indx:
            image_names = []
            if 'view' in conf:
                cam_ids = [conf['view']]
            else:
                if 'ncam' in conf:
                    ncam = conf['ncam']
                else: ncam = 1
                cam_ids = range(ncam)

            for icam in cam_ids:
                image_names.append(str(i) + '/image_view{}/encoded'.format(icam))
                features_name[image_names[-1]] = tf.FixedLenFeature([1], tf.string)

            if 'image_only' not in conf:
                action_name = str(i) + '/action'
                endeffector_pos_name = str(i) + '/endeffector_pos'


            if 'image_only' not in conf:
                features_name[action_name] = tf.FixedLenFeature([adim], tf.float32)
                features_name[endeffector_pos_name] = tf.FixedLenFeature([sdim], tf.float32)

            if 'test_metric' in conf:
                robot_pos_name = str(i) + '/robot_pos'
                object_pos_name = str(i) + '/object_pos'
                features_name[robot_pos_name] = tf.FixedLenFeature([conf['test_metric']['robot_pos'] * 2], tf.int64)
                features_name[object_pos_name] = tf.FixedLenFeature([conf['test_metric']['object_pos'] * 2], tf.int64)

            if 'load_vidpred_data' in conf:
                gen_image_name = str(i) + '/gen_images'
                gen_states_name = str(i) + '/gen_states'
                features_name[gen_image_name] = tf.FixedLenFeature([1], tf.string)
                features_name[gen_states_name] = tf.FixedLenFeature([sdim], tf.float32)


            features = tf.parse_single_example(serialized_example, features=features_name)

            images_t = []
            for image_name in image_names:
                image = decode_im(conf, features, image_name)

                if 'color_augmentation' in conf:
                    # print 'performing color augmentation'
                    image_hsv = tf.image.rgb_to_hsv(image)
                    img_stack = [tf.unstack(imag, axis=2) for imag in tf.unstack(image_hsv, axis=0)]
                    stack_mod = [tf.stack([x[0] + rand_h,
                                           x[1] + rand_s,
                                           x[2] + rand_v]
                                          , axis=2) for x in img_stack]

                    image_rgb = tf.image.hsv_to_rgb(tf.stack(stack_mod))
                    image = tf.clip_by_value(image_rgb, 0.0, 1.0)
                images_t.append(image)

            image_seq.append(tf.stack(images_t, axis=1))

            if 'image_only' not in conf:
                endeffector_pos = tf.reshape(features[endeffector_pos_name], shape=[1, sdim])
                endeffector_pos_seq.append(endeffector_pos)
                action = tf.reshape(features[action_name], shape=[1, adim])
                action_seq.append(action)

            if 'test_metric' in conf:
                robot_pos = tf.reshape(features[robot_pos_name], shape=[1, 2])
                robot_pos_seq.append(robot_pos)

                object_pos = tf.reshape(features[object_pos_name], shape=[1, conf['test_metric']['object_pos'], 2])
                object_pos_seq.append(object_pos)

            if 'load_vidpred_data' in conf:
                gen_images_seq.append(decode_im(gen_image_name))
                gen_states = tf.reshape(features[gen_states_name], shape=[1, sdim])
                gen_states_seq.append(gen_states)

        return_dict = {}
        image_seq = tf.concat(values=image_seq, axis=0)
        image_seq = tf.squeeze(image_seq)
        if 'use_cam' in conf:
            image_seq = image_seq[:,conf['use_cam']]
        return_dict['images'] = image_seq

        if 'goal_image' in conf:
            features_name = {}
            features_name['/goal_image'] = tf.FixedLenFeature([1], tf.string)
            features = tf.parse_single_example(serialized_example, features=features_name)
            goal_image = tf.squeeze(decode_im(conf, features, '/goal_image'))
            return_dict['goal_image'] = goal_image

        if 'first_last_noarm' in conf:
            features_name = {}
            features_name['/first_last_noarm0'] = tf.FixedLenFeature([1], tf.string)
            features = tf.parse_single_example(serialized_example, features=features_name)
            first_last_noarm0 = tf.squeeze(decode_im(conf, features, '/first_last_noarm0'))
            features_name['/first_last_noarm1'] = tf.FixedLenFeature([1], tf.string)
            features = tf.parse_single_example(serialized_example, features=features_name)
            first_last_noarm1 = tf.squeeze(decode_im(conf, features, '/first_last_noarm1'))
            return_dict['first_last_noarm'] = tf.stack([first_last_noarm0, first_last_noarm1], axis=0)

        if 'image_only' not in conf:
            if 'no_touch' in conf:
                return_dict['endeffector_pos'] = tf.concat(endeffector_pos_seq, 0)[:,:-2]
            else:
                return_dict['endeffector_pos'] = tf.concat(endeffector_pos_seq, 0)

            if 'autograsp' in conf:
                return_dict['actions'] = tf.concat(action_seq, 0)[:,:-1]
            else:
                return_dict['actions'] = tf.concat(action_seq, 0)

        if 'load_vidpred_data' in conf:
            return_dict['gen_images'] = gen_images_seq
            return_dict['gen_states'] = gen_states_seq

        return return_dict

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)

    if 'max_epoch' in conf:
        dataset = dataset.repeat(conf['max_epoch'])
    else: dataset = dataset.repeat()

    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffersize)
    dataset = dataset.batch(conf['batch_size'])
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    output_element = {}
    for k in list(next_element.keys()):
        output_element[k] = tf.reshape(next_element[k], [conf['batch_size']] + next_element[k].get_shape().as_list()[1:])

    return output_element