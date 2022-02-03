import tensorflow as tf


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return x


def leaky_relu(x):
    return tf.nn.leaky_relu(x)


def tanh(x):
    return tf.nn.tanh(x)


def make_wts_and_bias(weights, biases, input_channels, output_channels, type):
    curr_weights_num = len(weights)
    curr_biases_num = len(biases)
    if type == 'normal':
        new_w = 'wc' + str(curr_weights_num + 1)
        weights[new_w] = tf.get_variable('W' + str(curr_weights_num + 1), shape=(3, 3, input_channels, output_channels),
                                         initializer=tf.random_normal_initializer(0, 0.02))
        new_b = 'b' + str(curr_biases_num + 1)
        biases[new_b] = tf.get_variable('B' + str(curr_biases_num + 1), shape=(output_channels),
                                        initializer=tf.random_normal_initializer(0, 0.02))
    return weights, biases, new_w, new_b

def batch_norm(x, bns):
    mean, var = tf.nn.moments(x, axes= [1, 2], keep_dims= True)
    epsilon = 1e-8

    curr_bn = int(len(bns)/2)
    beta = tf.get_variable('beta' + str(curr_bn + 1), shape= (x.shape[3]), initializer= tf.initializers.zeros)
    gamma = tf.get_variable('gamma' + str(curr_bn + 1), shape= (x.shape[3]), initializer= tf.initializers.ones)
    bns.append(beta)
    bns.append(gamma)

    # beta = tf.get_variable(tf.zeros([x.shape[3]]))
    # gamma = tf.Variable(tf.ones([x.shape[3]]))

    x = tf.divide(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))
    x = gamma*x + beta

    return x


def conv_block(x, weights, biases, bns, num_filters, activation, use_batch_norm):
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, x.shape[3], num_filters, 'normal')
    x = conv2d(x, weights[new_w], biases[new_b])

    if use_batch_norm == 1:
        x = batch_norm(x, bns)

    if activation == 'leaky_relu':
        x = leaky_relu(x)
    elif activation == 'tanh':
        x = tanh(x)
    else:
        pass

    return x


def conv_net(x):
    weights = {}
    biases = {}
    bns = []

    conv1 = conv_block(x, weights, biases, bns, 32, 'leaky_relu', 1)
    conv2 = conv_block(conv1, weights, biases, bns, 32, 'leaky_relu', 1)
    conv3 = conv_block(conv2, weights, biases, bns,  32, 'leaky_relu', 1)
    conv4 = conv_block(conv3, weights, biases, bns, 32, 'leaky_relu', 1)
    conv5 = conv_block(conv4, weights, biases, bns,  1, 'none', 0)

    return conv5
