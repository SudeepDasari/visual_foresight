import tensorflow as tf
import numpy as np
import copy
from tensorflow.python.platform import gfile
from .vpred_model_interface import VPred_Model_Interface
from visual_mpc.utils.logger import Logger
from .checkpoint_matcher import variable_checkpoint_matcher
import re
from tensorflow.python.framework.errors_impl import NotFoundError


def get_maxiter_weights(dir):
    try:
        filenames = gfile.Glob(dir + '/model*')
    except NotFoundError:
        print('nothing found at ', dir + '/model*')
        return None
    iternums = []
    if len(filenames) != 0:
        for f in filenames:
            try:
                iternums.append(int(re.match('.*?([0-9]+)$', f).group(1)))
            except:
                iternums.append(-1)
        iternums = np.array(iternums)
        return filenames[np.argmax(iternums)].split('.')[0]  # skip the str after the '.'
    else:
        return None


class Tower(object):
    def __init__(self, conf, gpu_id, start_images, actions, start_states, pix_distrib):

        nsmp_per_gpu = conf['batch_size'] // conf['ngpu']
        # setting the per gpu batch_size

        # picking different subset of the actions for each gpu
        startidx = gpu_id * nsmp_per_gpu
        actions = tf.slice(actions, [startidx, 0, 0], [nsmp_per_gpu, -1, -1])
        start_images = tf.tile(start_images, [nsmp_per_gpu, 1, 1, 1, 1, 1])
        start_states = tf.tile(start_states, [nsmp_per_gpu, 1, 1])

        if pix_distrib is not None:
            pix_distrib = tf.tile(pix_distrib, [nsmp_per_gpu, 1, 1, 1, 1, 1])

        print('startindex for gpu {0}: {1}'.format(gpu_id, startidx))

        Model = conf['pred_model']
        print('using pred_model', Model)

        # this is to keep compatiblity with old model implementations (without basecls structure)
        if hasattr(Model, 'm'):
            for name, value in Model.m.__dict__.items():
                setattr(Model, name, value)

        modconf = copy.deepcopy(conf)
        modconf['batch_size'] = nsmp_per_gpu
        self.model = Model(modconf, start_images, actions, start_states, pix_distrib=pix_distrib, build_loss=False)


def setup_predictor(hyperparams, conf, gpu_id=0, ngpu=1, logger=None):
    """
    Setup up the network for control
    :param hyperparams: general hyperparams, can include control flags
    :param conf_file for network
    :param ngpu number of gpus to use
    :return: function which predicts a batch of whole trajectories
    conditioned on the actions
    """
    assert conf['batch_size'] % ngpu == 0, "ngpu should perfectly divide batch_size"

    conf['ngpu'] = ngpu
    if logger == None:
        logger = Logger(printout=True)

    if 'ncam' in conf:
        ncam = conf['ncam']
    else:
        ncam = 1


    logger.log('making graph')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    g_predictor = tf.Graph()
    logger.log('making session')
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True), graph=g_predictor)
    logger.log('done making session.')
    with sess.as_default():
        with g_predictor.as_default():
            logger.log('Constructing multi gpu model for control...')

            if 'float16' in conf:
                use_dtype = tf.float16
            else:
                use_dtype = tf.float32

            orig_size = conf['orig_size']
            images_pl = tf.placeholder(use_dtype, name='images',
                                       shape=(1, conf['context_frames'], ncam, orig_size[0], orig_size[1], 3))
            sdim = conf['sdim']
            adim = conf['adim']
            logger.log('adim', adim)
            logger.log('sdim', sdim)

            actions_pl = tf.placeholder(use_dtype, name='actions',
                                        shape=(conf['batch_size'], conf['sequence_length'], adim))
            states_pl = tf.placeholder(use_dtype, name='states',
                                       shape=(1, conf['context_frames'], sdim))

            if 'use_goal_image' in conf or 'no_pix_distrib' in conf:
                pix_distrib = None
            else:
                pix_distrib = tf.placeholder(use_dtype, shape=(
                1, conf['context_frames'], ncam, orig_size[0], orig_size[1], conf['ndesig']))

            # making the towers
            towers = []
            for i_gpu in range(gpu_id, ngpu + gpu_id):
                with tf.device('/device:GPU:{}'.format(i_gpu)):
                    with tf.name_scope('tower_%d' % (i_gpu)):
                        logger.log(('creating tower %d: in scope %s' % (i_gpu, tf.get_variable_scope())))
                        towers.append(Tower(conf, i_gpu, images_pl, actions_pl, states_pl, pix_distrib))
                        tf.get_variable_scope().reuse_variables()

            sess.run(tf.global_variables_initializer())

            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            vars = filter_vars(vars)

            if 'load_latest' in hyperparams:
                conf['pretrained_model'] = get_maxiter_weights('/result/modeldata')
                logger.log('loading {}'.format(conf['pretrained_model']))
                if conf['pred_model'] == VPred_Model_Interface:
                    towers[0].model.m.restore(sess, conf['pretrained_model'])
                else:
                    vars = variable_checkpoint_matcher(conf, vars, conf['pretrained_model'])
                    saver = tf.train.Saver(vars, max_to_keep=0)
                    saver.restore(sess, conf['pretrained_model'])
            else:
                if conf['pred_model'] == VPred_Model_Interface:
                    towers[0].model.m.restore(sess, conf['pretrained_model'])
                else:
                    vars = variable_checkpoint_matcher(conf, vars, conf['pretrained_model'])
                    saver = tf.train.Saver(vars, max_to_keep=0)
                    saver.restore(sess, conf['pretrained_model'])

            logger.log('restore done. ')

            logger.log('-------------------------------------------------------------------')
            logger.log('verify current settings!! ')
            for key in list(conf.keys()):
                logger.log(key, ': ', conf[key])
            logger.log('-------------------------------------------------------------------')

            comb_gen_img = tf.concat([to.model.gen_images for to in towers], axis=0)
            if towers[0].model.gen_states is not None:
                comb_gen_states = tf.concat([to.model.gen_states for to in towers], axis=0)
            else:
                comb_gen_states = None

            if not 'no_pix_distrib' in conf:
                comb_pix_distrib = tf.concat([to.model.gen_distrib for to in towers], axis=0)

            def predictor_func(input_images=None, input_one_hot_images=None, input_state=None, input_actions=None):
                """
                :param one_hot_images: the first two frames
                :param pixcoord: the coords of the disgnated pixel in images coord system
                :return: the predicted pixcoord at the end of sequence
                """

                feed_dict = {}
                for t in towers:
                    if hasattr(t.model, 'iter_num'):
                        feed_dict[t.model.iter_num] = 0

                feed_dict[images_pl] = input_images
                feed_dict[states_pl] = input_state
                feed_dict[actions_pl] = input_actions

                if input_one_hot_images is None:
                    if comb_gen_states is None:
                        gen_images = sess.run(comb_gen_img, feed_dict)
                        gen_states = None
                    else:
                        gen_images, gen_states = sess.run([comb_gen_img,
                                                       comb_gen_states],
                                                      feed_dict)
                    gen_distrib = None
                elif comb_gen_states is None:
                    feed_dict[pix_distrib] = input_one_hot_images
                    gen_images, gen_distrib = sess.run([comb_gen_img, comb_pix_distrib], feed_dict)
                    gen_states = None
                else:
                    feed_dict[pix_distrib] = input_one_hot_images
                    gen_images, gen_distrib, gen_states = sess.run([comb_gen_img,
                                                                    comb_pix_distrib,
                                                                    comb_gen_states],
                                                                   feed_dict)

                return gen_images, gen_distrib, gen_states

            return predictor_func


def filter_vars(vars):
    newlist = []
    for v in vars:
        if not '/state:' in v.name:
            newlist.append(v)
        else:
            print('removed state variable from saving-list: ', v.name)

    return newlist
