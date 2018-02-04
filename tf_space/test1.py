# -*- coding: utf-8 -*-

import tensorflow as tf

SEED = 1
tf.set_random_seed(SEED)

w1 = tf.Variable(tf.random_normal([2,3], stddev=2))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1))

x = tf.constant([[0.7, 0.9]])

a = tf.matmul(x, w1)
