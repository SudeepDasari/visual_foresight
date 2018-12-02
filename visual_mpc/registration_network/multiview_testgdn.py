import tensorflow as tf
import matplotlib; matplotlib.use('Agg');

from visual_mpc.video_prediction.checkpoint_matcher import variable_checkpoint_matcher
from .gdnet import GoalDistanceNet


def mean_square(x):
    return tf.reduce_mean(tf.square(x))

def length(x):
    return tf.sqrt(tf.reduce_sum(tf.square(x), 3))

class MulltiviewTestGDN():
    def __init__(self,
                 conf = None,
                 build_loss=False,
                 load_data=False
                 ):
        self.conf = conf
        self.gdn = []

        self.img_height = conf['orig_size'][0]
        self.img_width = conf['orig_size'][1]
        self.ncam = len(conf['pretrained_model'])

        self.I0_pl = tf.placeholder(tf.float32, name='I0', shape=(conf['batch_size'], self.ncam, self.img_height, self.img_width, 3))
        self.I1_pl = tf.placeholder(tf.float32, name='I1', shape=(conf['batch_size'], self.ncam, self.img_height, self.img_width, 3))

        if 'model' in conf:
            Model = conf['model']
        else:
            Model = GoalDistanceNet

        for n in range(self.ncam):
            self.gdn.append(Model(conf=conf, build_loss=False, load_data = False,
                                            I0=self.I0_pl[:,n], I1=self.I1_pl[:,n]))

    def build_net(self):
        self.warped_I0_to_I1 = []
        self.flow_bwd = []
        self.warp_pts_bwd = []

        self.scopenames = []
        for n in range(self.ncam):
            name = "gdncam{}".format(n)
            print('bulding ', name)
            with tf.variable_scope(name):
                self.gdn[n].build_net()

            self.warped_I0_to_I1.append(self.gdn[n].warped_I0_to_I1)
            self.flow_bwd.append(self.gdn[n].flow_bwd)
            self.warp_pts_bwd.append(self.gdn[n].warp_pts_bwd)
            self.scopenames.append(name)
        self.warped_I0_to_I1 = tf.stack(self.warped_I0_to_I1, axis=1)
        self.flow_bwd = tf.stack(self.flow_bwd, axis=1)
        self.warp_pts_bwd = tf.stack(self.warp_pts_bwd, axis=1)

    def restore(self, sess):
        for n in range(self.ncam):
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scopenames[n])
            modelfile = self.conf['pretrained_model'][n]
            vars = variable_checkpoint_matcher(self.conf, vars, modelfile, ignore_varname_firstag=True)
            saver = tf.train.Saver(vars, max_to_keep=0)
            saver.restore(sess, modelfile)
            print('gdn{} restore done.'.format(n))


