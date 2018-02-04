# -*- coding:utf-8 -*-

import tensorflow as tf

INPUT_NODE = 28 * 28
OUTPUT_NODE = 10

LAYER1_NODE = 500

VAR_SCOPE = 'nn_mnist'

MNIST_DATA_PATH = 'MNIST_data'

MODEL_SAVE_PATH = 'MNIST_model'
MODEL_NAME = 'nn_mnist.ckpt'

def get_weight_variable(reuse=True):
    with tf.variable_scope(VAR_SCOPE, reuse=reuse):
        layer1_weights = tf.get_variable('layer1_weights', [INPUT_NODE, LAYER1_NODE],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer1_bias = tf.get_variable('layer1_bias', [LAYER1_NODE,],
            initializer=tf.constant_initializer(0.1))

        layer2_weights = tf.get_variable('layer2_weights', [LAYER1_NODE, OUTPUT_NODE],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer2_bias = tf.get_variable('layer2_bias', [OUTPUT_NODE,],
            initializer=tf.constant_initializer(0.1))

    return (layer1_weights, layer1_bias, layer2_weights, layer2_bias)

def create_weight_variable():
    return get_weight_variable(reuse=False)

def inference(input_tensor):
    layer1_weights, layer1_bias, layer2_weights, layer2_bias = get_weight_variable()
    
    layer1 = tf.nn.relu(tf.matmul(input_tensor, layer1_weights) + layer1_bias)
    output_tensor = tf.matmul(layer1, layer2_weights) + layer2_bias

    return output_tensor

def load_data():
    from tensorflow.examples.tutorials import mnist
    mnist_data = mnist.input_data.read_data_sets(MNIST_DATA_PATH, one_hot=True)
    return mnist_data


