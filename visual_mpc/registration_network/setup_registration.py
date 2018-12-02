import tensorflow as tf
import os
from .multiview_testgdn import MulltiviewTestGDN

def setup_gdn(conf, gpu_id = 0):
    """
    Setup up the network for control
    :param conf_file:
    :return: function which predicts a batch of whole trajectories
    conditioned on the actions
    """
    if gpu_id == None:
        gpu_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print('using CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"])

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    g_predictor = tf.Graph()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph= g_predictor)
    with sess.as_default():
        with g_predictor.as_default():
            print('Constructing model Warping Network')
            model = MulltiviewTestGDN(conf=conf,
                          build_loss=False,
                          load_data = False)
            model.build_net()
            model.restore(sess)

            def predictor_func(pred_images, goal_images):
                feed_dict = {
                            model.I0_pl:pred_images,
                            model.I1_pl:goal_images}

                warped_images, flow_field, warp_pts = sess.run([model.warped_I0_to_I1,
                                                                model.flow_bwd,
                                                                model.warp_pts_bwd],
                                                               feed_dict)
                return warped_images, flow_field, warp_pts

            return predictor_func