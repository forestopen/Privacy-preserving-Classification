# -*- coding: utf-8 -*-
'''
把图片用RBV处理并展示原始图片，RBV图片，RBVDP图片

'''
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import encoding as en
import rbv.RandomizedBitVector as rbv

if __name__ == "__main__":
    p = 0.5
    rndList = rbv.generateRandomVariable(lower=0, upper=255, length=8)
    rndList = [235, 106, 177, 222, 23, 113, 112, 197]
    print("rndList = " + str(rndList))

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    batch_size = 1
    n_batch = mnist.train.num_examples // batch_size

    # 第一张图
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    x = np.asarray(batch_xs[0])
    x = x * 255
    x1 = x.astype(np.uint8)
    x2 = en.listToRBVList(x1, rndList)
    x3 = en.listToRBVListwithDP(x1, rndList, p)

    print(x1.reshape(28, 28))
    print(x2.reshape(28, 28))
    print(x3.reshape(28, 28))
    cv2.imshow("test1", x1.reshape(28, 28))
    cv2.imshow("test2", x2.reshape(28, 28))
    cv2.imshow("test3", x3.reshape(28, 28))
    cv2.waitKey(0)
