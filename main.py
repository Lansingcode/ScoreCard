# -*- coding:utf-8 -*-
__author__ = 'xujia'

import pandas as pd


def fileInfo(path):
    '''
    获取文件信息
    :param path: 文件路径
    :return: {字段名称：[字段类型，数据量，空值数]}
    '''
    infodict = {}
    data = pd.read_csv(path)
    for c in data.columns:
        infodict[c] = data[c].dtype
        ctype = data[c].dtype
        nc = data[c].size - data[c].notnull().sum()
        infodict[c] = [ctype, data[c].size, nc]  # 字段类型，数据量，空值个数
    return infodict


if __name__ == '__main__':
    # path=input('Please input the file path: ')
    path = 'iris.csv'
    ret = fileInfo(path)
