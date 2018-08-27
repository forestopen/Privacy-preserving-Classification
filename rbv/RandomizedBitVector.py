# -*- coding: utf-8 -*-
import random
import numpy as np


def randomizedbitvector(r, x):
    ar = np.zeros(len(r), int)
    ar[np.where(x >= r)] = 1
    return ar


# 生成长为length的随机数字
def generateRandomVariable(lower, upper, length):
    return [random.randint(lower, upper) for i in range(length)]


if __name__ == "__main__":
    r = np.asarray([15, 20])
    x = 18
    print(randomizedbitvector(r, x))
    print(generateRandomVariable(lower=5, upper=10, length=5))