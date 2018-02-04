# -*- coding:utf-8 -*-

import tensorflow as tf

import nn_mnist
reload(nn_mnist)

def evaluate_core(mnist_data):
    with tf.Graph().as_default() as g:
        X = tf.placeholder(tf.float32, [None,nn_mnist.INPUT_NODE], name='X')
        y_ = tf.placeholder(tf.float32, [None,nn_mnist.OUTPUT_NODE], name='y_')

        validate_feed = {X: mnist_data.validation.images, y_: mnist_data.validation.labels}

        nn_mnist.create_weight_variable()
        y = nn_mnist.inference(X)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(nn_mnist.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                training_steps = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print('After %s training steps, validation accuracy = %g.' % (training_steps, accuracy_score))

def evaluate():
    mnist_data = nn_mnist.load_data()
    evaluate_core(mnist_data)

if '__main__' == __name__:
    evaluate()
