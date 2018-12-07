import numpy as np
import tensorflow as tf
from visual_mpc.datasets.base_dataset import BaseVideoDataset


dataset_path = None                              # path to dataset (aka './towel_pick_30k')  
dataset = BaseVideoDataset(dataset_path, 16)     # parses tfrecords automatically

img_tensor = dataset['images', 'train']          # (batch_size, T, height, width, 3)   
# 'train' is default mode if ommited
actions = dataset['actions']                     # (batch_size, T, adim)

sess = tf.Session()
print(sess.run(actions).shape)
print(sess.run(img_tensor).shape)
