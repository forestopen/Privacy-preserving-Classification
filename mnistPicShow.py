# -*- coding: utf-8 -*-
import tensorflow as tf
import cv2
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

batch_size = 1
n_batch = mnist.train.num_examples // batch_size

# 第一张图
batch_xs, batch_ys = mnist.train.next_batch(batch_size)
print(batch_xs.shape, batch_ys)

fig = np.asarray(batch_xs[0])
fig = fig * 255
print(fig)
fig_uint8 = fig.astype(np.uint8).reshape(28,28)
cv2.imshow('test', fig_uint8)
cv2.waitKey(0)

# for batch in range(n_batch):
#     batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#     print(batch_xs, batch_ys)
