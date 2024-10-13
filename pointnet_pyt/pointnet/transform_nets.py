import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util


def input_transform_net(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = tf.shape(point_cloud)[0]  # Get the batch size dynamically
    num_point = tf.shape(point_cloud)[1]   # Get the number of points dynamically

    # Expand dimensions of point cloud to [batch_size, num_point, 3, 1]
    input_image = tf.expand_dims(point_cloud, -1)
    print(f"Shape of input_image before conv2d: {input_image.shape}")

    # Use conv2d with kernel size [1, 1] to process each point independently
    net = tf_util.conv2d(input_image, 64, [1, 1],  # Change kernel size to [1, 1]
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    
    # Continue with pointwise convolution using [1, 1] kernels
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    
    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)

    # Max pooling over all points to extract global features
    net = tf_util.max_pool2d(net, [num_point, 1],
                             padding='VALID', scope='tmaxpool')

    # Flatten the pooled feature map and pass through fully connected layers
    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.name_scope('transform_XYZ'):
        assert K == 3
        weights = tf.Variable(tf.zeros([256, 3 * K]), name='weights')
        biases = tf.Variable(tf.zeros([3 * K]), name='biases')
        biases = biases + tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    # Reshape the final transform matrix
    transform = tf.reshape(transform, [batch_size, 3, K])
    return transform



def feature_transform_net(inputs, is_training, bn_decay=None, K=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """
    batch_size = tf.shape(inputs)[0]  # Get batch size dynamically
    num_point = tf.shape(inputs)[1]   # Get num points dynamically

    net = tf_util.conv2d(inputs, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point, 1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.name_scope('transform_feat'):
        weights = tf.Variable(tf.zeros([256, K * K]), name='weights')
        biases = tf.Variable(tf.zeros([K * K]), name='biases')
        biases = biases + tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform
