# -*- coding: utf-8 -*-
'''
把MNIST数据集中的文件都保存下来
'''
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cv2

# 返回图片类别 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] 返回 5
def getLabel(arr):
    for i in range(len(arr)):
        if arr[i] > 0.5:
            return i
    print("Error in getLabel: label not founded!")
    exit(0)


if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    batch_size = 1
    n_batch = mnist.train.num_examples // batch_size
    print(str(n_batch) + " pictures founded in mnist.train")

    f = open('data/mnist_train_data.txt', 'w')

    for batch in range(n_batch):
        print(batch)
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        x0 = np.asarray(batch_xs[0]) * 255
        x1 = x0.astype(np.uint8)

        x_print = ' '.join(str(ele) for ele in x1)
        y_print = str(getLabel(batch_ys[0]))
        str_print = x_print + "-" + y_print + "\n"
        f.write(str_print)
    f.close()