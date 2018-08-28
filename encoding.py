# -*- coding: utf-8 -*-
'''
把图片用RBV处理

'''
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import rbv.RandomizedBitVector as rbv

'''
把一个0-255的数转换为rbv，再将RBV编码变回0-255的数
'''
def valueToRBVValue(val, rndList):
    vTemp = rbv.randomizedbitvector(rndList=rndList, value=val)
    strTemp = ''.join(str(a) for a in vTemp.tolist())
    value = int(strTemp, 2)
    return value


def valueToRBVValuewithDP(val, rndList, p):
    vTemp = rbv.randomizedbitvectorwithDP(rndList=rndList, value=val, p=p)
    strTemp = ''.join(str(a) for a in vTemp.tolist())
    value = int(strTemp, 2)
    return value


def listToRBVList(inputList, rndList):
    rbvList = np.asarray([valueToRBVValue(pixel, rndList) for pixel in inputList]).astype(np.uint8)
    return rbvList


def listToRBVListwithDP(inputList, rndList, p):
    rbvList = np.asarray([valueToRBVValuewithDP(pixel, rndList, p) for pixel in inputList]).astype(np.uint8)
    return rbvList
