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
        ctype = data[c].dtype
        nc = data[c].size - data[c].notnull().sum()
        infodict[c] = [ctype, data[c].size, nc]  # 字段类型，数据量，空值个数
    return infodict, data


def changeType(df, featypedict):
    typedict = {1: 'float64', 2: 'int64', 3: 'str'}
    feadict = dict(zip(list(range(df.shape[1])), df.columns.values))

    print('当前数据类型为：')
    for (k, v) in featypedict.items():
        print(k.rjust(15), v[0])
    print('字段名称对应数字为：')
    for (n, m) in feadict.items():
        print(n, m)
    feaName = input('请输入如需要更改数据类型的字段对应的数字：')
    if int(feaName) not in feadict.keys():
        feaName = input('输入字段名称错误，请重新输入：')
        if int(feaName) not in feadict.keys():
            pass
    feaName = feadict[int(feaName)]

    type = input('请输入目标类型对应的数字(1: 浮点型(float64)，2: 整型(int64)，3: 字符型(str)：')
    if int(type) not in typedict.keys():
        type = input('请输入目标类型对应的数字(1: 浮点型(float64)，2: 整型(int64)，3: 字符型(str)：')
        if int(type) not in typedict.keys():
            pass
    type = typedict[int(type)]

    df[feaName] = df[feaName].astype(type)


def dataSplit(data):
    '''
    数据分割
    :param data:带分割数据
    :param ratio: 分割比例
    :return: （数据集1，数据集2）
    '''
    ratio = float(input('请输入数据分割比例：'))
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
    feadict, data = fileInfo(path)

    changeType(data, feadict)
    print(data.dtypes)

    t = dataSplit(data)
    print(t[0].shape)
    print(t[1].shape)
