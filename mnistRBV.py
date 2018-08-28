# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import encoding as en
import numpy as np
import cv2

def generateNewBatch(batch_xs, rndList, p):
    ret = []
    for x in batch_xs:
        x1 = x * 255
        x1 = x1.astype(np.uint8)
        # x2 = en.listToRBVList(x1, rndList)
        x3 = en.listToRBVListwithDP(x1, rndList, p)
        x4 = x3 / 255.0
        print(x1)
        print(x4)
        print("hehe")
        input()
        ret.append(x4)
    return ret


if __name__ == "__main__":

    rndList = [235, 106, 177, 222, 23, 113, 112, 197]
    p = 1

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    batch_size = 100
    n_batch = mnist.train.num_examples // batch_size

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    prediction = tf.nn.softmax(tf.matmul(x, W) + b)

    loss = tf.reduce_mean(tf.square(y-prediction))
    tran_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

    init = tf.global_variables_initializer()
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(21):
            print("in a epoch")
            batchFlag = 0
            for batch in range(n_batch):
                # print("in a batch : " + str(batchFlag))
                batchFlag += 1
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # print("in a batch")
                batch_new = generateNewBatch(batch_xs, rndList, p)
                sess.run(tran_step, feed_dict={x: batch_new, y: batch_ys})

            acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print("Iteration:" + str(epoch) + ", Testing Accuracy:" + str(acc))

