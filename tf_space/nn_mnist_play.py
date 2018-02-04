# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import nn_mnist
reload(nn_mnist)

mnist_data = nn_mnist.load_data()


sess = tf.InteractiveSession()

def load_nn(sess):
    X = tf.placeholder(tf.float32, [None,nn_mnist.INPUT_NODE], name='X')

    nn_mnist.get_weight_variable(reuse=tf.AUTO_REUSE)
    y = nn_mnist.inference(X)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(nn_mnist.MODEL_SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    return (X, y)


X, y = load_nn(sess)


def plot(img):
    plt.imshow(img.reshape([28,28]), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

def predict(sess, img, X, y):
    xs = np.array([img,])
    ys = tf.argmax(y, 1)
    result = sess.run(ys, feed_dict={X: xs})
    return result[0]

def play(idx):
    img = mnist_data.test.images[idx]
    plot(img)

    label = np.argmax(mnist_data.test.labels[idx], 0)
    result = predict(sess, img, X, y)

    print('label: %d, predict: %d' % (label, result))


