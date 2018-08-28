# -*- coding: utf-8 -*-
import random
import numpy as np


def randomizedbitvector(rndList, value):
    ar = np.zeros(len(rndList), int)
    ar[np.where(rndList <= value)] = 1
    return ar


def randomizedbitvectorwithDP(rndList, value, p):
    ar = np.zeros(len(rndList), int)
    ar[np.where(rndList <= value)] = 1
    p = (1.0 + p)/2
    change = np.random.binomial(1, p, len(rndList))
    return np.where(change == 1, ar, 1-ar)


# 生成长为length的随机数字
def generateRandomVariable(lower, upper, length):
    return np.asarray([random.randint(lower, upper) for i in range(length)])


if __name__ == "__main__":
    value = 15
    rndList = np.asarray([10, 14, 16, 20])
    p = 1
    print(randomizedbitvectorwithDP(rndList=rndList, value=value, p=p))
    print(randomizedbitvector(rndList=rndList, value=value))