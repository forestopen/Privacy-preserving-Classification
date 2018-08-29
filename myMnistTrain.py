# -*- coding: utf-8 -*-
import tensorflow as tf
import reader
import numpy as np


class mnist():
    def __init__(self, normalize=True):
        self.dataList, labelList = reader.readMnistData()
        self.labelList = self.one_hot(labelList)
        self.num_data = len(labelList)
        self.__now_pointer = 0

        if normalize:
            self.dataList = self.dataList / 255

    def next_batch(self, size):
        ret_list = []
        for i in range(size):
            ret_list.append(self.dataList[self.__now_pointer])
            self.__now_pointer = (self.__now_pointer + 1) % self.num_data
        return ret_list

    def one_hot(self, label_list):
        ret_list = []
        for label in label_list:
            ret_label = np.zeros(10)
            ret_label[int(label)] = 1
            ret_list.append(ret_label)
        return ret_list


if __name__ == "__main__":
    mni = mnist()
    print(mni.dataList[0])
    print(mni.num_data)
