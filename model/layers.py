import tensorflow as tf
'''
Layers for ResNet50 model
'''

def conv_layer(tensor, kernel_size, in_channel, out_channel, stride, name):
    '''
    2D Convolutional layer in TF
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable(
            "weights",
            shape=[kernel_size, kernel_size, in_channel, out_channel],
            initializer=tf.random_normal_initializer(stddev=0.02)
        )
        bias = tf.get_variable(
            "bias",
            shape=[out_channel],
            initializer=tf.constant_initializer(0.0)
        )
        conv = tf.nn.conv2d(
            tensor,
            weights,
            strides=[1, stride, stride, 1],
            padding="SAME"
        )
        conv = tf.nn.bias_add(conv, bias)
        return conv


def fc_layer(tensor, in_dims, out_dims, name):
    '''
    Fully-Connected layer in TF
    '''
    tensor = tf.reshape(tensor, shape=[-1, tensor.get_shape().as_list()[-1]])
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable(
            "weights",
            shape=[in_dims, out_dims],
            initializer=tf.random_normal_initializer(stddev=0.02)
        )
        bias = tf.get_variable(
            "bias",
            shape=[out_dims],
            initializer=tf.constant_initializer(0.0)
        )
        fc = tf.nn.bias_add(tf.matmul(tensor, weights), bias)
        return fc


def bn(tensor, is_training, name):
    '''
    Batch Normalization in TF
    Batch Normalization is not apllied is the model is training
    '''
    return tf.layers.batch_normalization(tensor, training=is_training, name=name)


def relu(tensor):
    '''
    Relu func in TF
    '''
    return tf.nn.relu(tensor)


def softmax(tensor):
    '''
    Softmax in TF
    '''
    return tf.nn.softmax(tensor)


def avgpool(tensor, kernel_size=2, stride=2, name="avg"):
    '''
    Average Pooling in TF
    '''
    return tf.nn.avg_pool(
        tensor,
        ksize=[1, kernel_size, kernel_size, 1],
        strides=[1, stride, stride, 1],
        padding="VALID",
        name=name
    )


def maxpool(tensor, kernel_size=2, stride=2, name="max"):
    '''
    Max Pooling in TF
    '''
    return tf.nn.max_pool(
        tensor,
        ksize=[1, kernel_size, kernel_size, 1],
        strides=[1, stride, stride, 1],
        padding="VALID",
        name=name
    )


def res_block_3_layer(tensor, channel_list, name, change_dimension, block_stride, is_training):
    '''
    Residual block for ResNet50
    '''
    with tf.variable_scope(name) as scope:
        if change_dimension:
            short_cut_conv = conv_layer(
                tensor,
                kernel_size=1,
                in_channel=tensor.get_shape().as_list()[-1],
                out_channel=channel_list[2],
                stride=block_stride,
                name="shortcut"
            )
            block_conv_input = bn(short_cut_conv, is_training, name="shortcut")
        else:
            block_conv_input = tensor

        block_conv1 = conv_layer(
            tensor,
            kernel_size=1,
            in_channel=tensor.get_shape().as_list()[-1],
            out_channel=channel_list[0],
            stride=block_stride,
            name="a"
        )
        block_conv1 = bn(block_conv1, is_training, name="a")
        block_conv1 = relu(block_conv1)

        block_conv2 = conv_layer(
            block_conv1,
            kernel_size=3,
            in_channel=channel_list[0],
            out_channel=channel_list[1],
            stride=1,
            name="b"
        )
        block_conv2 = bn(block_conv2, is_training, name="b")
        block_conv2 = relu(block_conv2)

        block_conv3 = conv_layer(
            block_conv2,
            kernel_size=1,
            in_channel=channel_list[1],
            out_channel=channel_list[2],
            stride=1,
            name="c"
        )
        block_conv3 = bn(block_conv3, is_training, name="c")

        block_res = tf.add(block_conv_input, block_conv3)
        res = relu(block_res)
        return res
