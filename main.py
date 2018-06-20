# -*- coding:utf-8 -*-
__author__ = 'xujia'

import pandas as pd
import numpy as np


def fileInfo(path):
    '''
    获取文件信息
    :param path: 文件路径
    :return: {字段名称：[字段类型，数据量，空值个数]}
    '''
    infodict = {}
    data = pd.read_csv(path)
    for c in data.columns:
        infodict[c] = data[c].dtype
        ctype = data[c].dtype
        nc = data[c].size - data[c].notnull().sum()
        infodict[c] = [ctype, data[c].size, nc]  # 字段类型，数据量，空值个数
    return infodict, data


def dataSplit(data, ratio):
    '''
    数据分割
    :param data:带分割数据
    :param ratio: 分割比例
    :return: （数据集1，数据集2）
    '''
    dataCount = data.shape[0]
    selectedCount = int(dataCount * ratio)
    if selectedCount > 0:
        splitedData = np.split(data.sample(frac=1), [selectedCount], axis=0)
    else:
        return 'Data is too less'
    return splitedData


if __name__ == '__main__':
    # path=input('Please input the file path: ')
    path = 'iris.csv'
    dict, data = fileInfo(path)
    t = dataSplit(data, 0.8)
    print(t[0])
    print(t[1])
