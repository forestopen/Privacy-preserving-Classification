# -*- coding: utf-8 -*-
'''
reader for mnist_train_data.txt

'''
import encoding as en
import rbv.RandomizedBitVector as rbv
import cv2
import numpy as np


def dataExtra(srcDataList):
    dataList = []
    labelList = []

    for line in srcDataList:
        tData = line.split('-')[0]
        tLabel = line.split('-')[1]

        dataList.append(np.asarray([int(x) for x in tData.split(' ')]))
        labelList.append(int(tLabel))
    return dataList, labelList

def readMnistData():
    print("start read mnist data")
    filename = 'data/mnist_train_data.txt'
    with open(filename, 'r') as f:
        srcDataList = f.readlines()
    dataList, labelList = dataExtra(srcDataList)
    print("read mnist data successful")
    return np.asarray(dataList), np.asarray(labelList)

def readRbvMnistData():
    filename = 'data/rbv_mnist_train_data.txt'
    with open(filename, 'r') as f:
        srcDataList = f.readlines()
    dataList, labelList = dataExtra(srcDataList)
    return dataList, labelList

def readRbvMnistData():
    filename = 'data/rbv_mnist_train_data.txt'
    with open(filename, 'r') as f:
        srcDataList = f.readlines()
    dataList, labelList = dataExtra(srcDataList)
    return np.asarray(dataList), np.asarray(labelList)

def writeRbvMnistData():
    print("start write rbvMnistData")
    rndList = rbv.generateRandomVariable(0, 255, 8)
    p = 1
    filename = 'data/rbv_mnist_train_data.txt'
    dataList, labelList = readMnistData()
    with open(filename, 'w') as f:
        for i in range(len(dataList)):
            print("processing data: " + str(i))
            data = dataList[i]
            rbvData = en.listToRBVListwithDP(data, rndList, p)
            rbvDataStr = ' '.join(str(value) for value in rbvData)
            f.write(rbvDataStr + '-' + str(labelList[i]) + "\n")
    print("rbvMnistData write successful")

if __name__ == "__main__":
    writeRbvMnistData()
    # dataList, labelList = readRbvMnistData()
    # print(len(dataList))
