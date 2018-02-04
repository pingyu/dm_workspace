# -*- coding:utf-8 -*-

import tensorflow as tf
import os

import nn_mnist
reload(nn_mnist)

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

REGULARATION_RATE = 0.0001

TRAINING_STEPS = 30000

def train_core(mnist_data):
    ### define flow ###
    X = tf.placeholder(tf.float32, [None,nn_mnist.INPUT_NODE], name='X')
    y_ = tf.placeholder(tf.float32, [None,nn_mnist.OUTPUT_NODE], name='y_')

    layer1_weights, layer1_bias, layer2_weights, layer2_bias = nn_mnist.create_weight_variable()
    y = nn_mnist.inference(X)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    regularizer = tf.contrib.layers.l2_regularizer(REGULARATION_RATE)
    loss = tf.reduce_mean(cross_entropy) + regularizer(layer1_weights) + regularizer(layer2_weights)

    ### train step ###
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        learning_rate = LEARNING_RATE_BASE,
        global_step = global_step,
        decay_steps = mnist_data.train.num_examples / BATCH_SIZE,
        decay_rate = LEARNING_RATE_DECAY,
        staircase = True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ### do train ###
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in xrange(TRAINING_STEPS):
            xs, ys = mnist_data.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_step, loss, global_step], feed_dict={X:xs, y_:ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss is %g." % (step, loss_value))
                saver.save(sess, os.path.join(nn_mnist.MODEL_SAVE_PATH, nn_mnist.MODEL_NAME), global_step=global_step)

        print("FINAL: After %d training step(s), loss is %g." % (step, loss_value))
        saver.save(sess, os.path.join(nn_mnist.MODEL_SAVE_PATH, nn_mnist.MODEL_NAME), global_step=global_step)


def train():
    mnist_data = nn_mnist.load_data()
    train_core(mnist_data)


if '__main__' == __name__:
    train()

