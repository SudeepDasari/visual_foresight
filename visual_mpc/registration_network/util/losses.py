import tensorflow as tf
import numpy as np


def create_mask(tensor, paddings):
    with tf.variable_scope('create_mask'):
        shape = tf.shape(tensor)
        inner_width = shape[1] - (paddings[0][0] + paddings[0][1])
        inner_height = shape[2] - (paddings[1][0] + paddings[1][1])
        inner = tf.ones([inner_width, inner_height])

        mask2d = tf.pad(inner, paddings)
        mask3d = tf.tile(tf.expand_dims(mask2d, 0), [shape[0], 1, 1])
        mask4d = tf.expand_dims(mask3d, 3)
        return tf.stop_gradient(mask4d)


def charbonnier_loss(x, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):
    """Compute the generalized charbonnier loss of the difference tensor x.
    All positions where mask == 0 are not taken into account.

    Args:
        x: a tensor of shape [num_batch, height, width, channels].
        mask: a mask of shape [num_batch, height, width, mask_channels],
            where mask channels must be either 1 or the same number as
            the number of channels of x. Entries should be 0 or 1.
    Returns:
        loss as tf.float32
    """
    with tf.variable_scope('charbonnier_loss'):
        batch, height, width, channels = tf.unstack(tf.shape(x))
        normalization = tf.cast(batch * height * width * channels, tf.float32)

        error = tf.pow(tf.square(x * beta) + tf.square(epsilon), alpha)

        if mask is not None:
            error = tf.multiply(mask, error)

        if truncate is not None:
            error = tf.minimum(error, truncate)

        return tf.reduce_sum(error) / (normalization + 1e-6)


def ternary_loss(im1, im2_warped, mask, max_distance=1):
    patch_size = 2 * max_distance + 1
    with tf.variable_scope('ternary_loss'):
        def _ternary_transform(image):
            intensities = tf.image.rgb_to_grayscale(image) * 255
            # patches = tf.extract_image_patches( # fix rows_in is None
            #    intensities,
            #    ksizes=[1, patch_size, patch_size, 1],
            #    strides=[1, 1, 1, 1],
            #    rates=[1, 1, 1, 1],
            #    padding='SAME')
            out_channels = patch_size * patch_size
            w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
            weights = tf.constant(w, dtype=tf.float32)
            patches = tf.nn.conv2d(intensities, weights, strides=[1, 1, 1, 1], padding='SAME')

            transf = patches - intensities
            transf_norm = transf / tf.sqrt(0.81 + tf.square(transf))
            return transf_norm

        def _hamming_distance(t1, t2):
            dist = tf.square(t1 - t2)
            dist_norm = dist / (0.1 + dist)
            dist_sum = tf.reduce_sum(dist_norm, 3, keep_dims=True)
            return dist_sum

        t1 = _ternary_transform(im1)
        t2 = _ternary_transform(im2_warped)
        dist = _hamming_distance(t1, t2)

        transform_mask = create_mask(mask, [[max_distance, max_distance],
                                            [max_distance, max_distance]])
        return charbonnier_loss(dist, mask * transform_mask)