# -*- coding:utf-8 -*-
__author__ = 'xujia'

import pandas as pd
import numpy as np
from scipy import stats


def equal_distance_binning(df, fea_name, target_name, bin_count):
    """
    等距分箱
    :param df:
    :param fea_name:
    :param bin_count
    :return:
    """
    df[fea_name + '_d'] = pd.cut(df[fea_name], bin_count)
    fea_count = df[[fea_name + '_d', target_name]].copy().groupby(
        [fea_name + '_d', target_name]).size().unstack().fillna(0.0)
    return fea_count


def equal_frequency_binning(df, fea_name, target_name, bin_count):
    """
    等频分箱
    :param df:
    :param fea_name:
    :param bin_count
    :return:
    """
    df[fea_name + '_f'] = pd.cut(df[fea_name], bin_count)
    fea_count = df[[fea_name + '_f', target_name]].copy().groupby(
        [fea_name + '_f', target_name]).size().unstack().fillna(0.0)
    return fea_count


def auto_binning(df, fea_name, target_name, max_bin_count):
    """
    自动分箱
    :param df:
    :param target_name: 目标变量名
    :param fea_name:特征变量名称
    :param max_bin_count:最大分箱数
    :return:
    """
    r = 0
    while np.abs(r) < 1:
        d1 = pd.DataFrame({'X': df[fea_name],
                           'Y': df[target_name],
                           fea_name + '_d': pd.qcut(df[fea_name], max_bin_count, duplicates='drop')})
        d2 = d1.groupby(fea_name + '_d', as_index=True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        max_bin_count = max_bin_count - 1

    fea_count = df[[fea_name + '_d', target_name]].copy().groupby(
        [fea_name + '_d', target_name]).size().unstack().fillna(0.0)
    return fea_count


def chi2(A):
    """
    计算卡方值
    :param A:需要计算卡方的两行数据
    :return: 卡方值
    """
    m, k = A.shape  # 行数 列数

    R = A.sum(axis=1)  # 行求和结果
    C = A.sum(axis=0)  # 列求和结果
    N = A.sum()  # 总和

    res = 0
    for i in range(m):
        for j in range(k):
            Eij = 1.0 * R[i] * C[j] / N
            if Eij != 0:
                res = 1.0 * res + (A[i][j] - Eij) ** 2 / Eij
    return res


def chi_merge(df, fea_name, target_name, max_bin_count):
    """
    chiMerge的主算法
    :param df:数据，dataframe格式
    :param fea_name:需要进行分段的特征名称
    :param target_name:目标变量名称
    :param dis_count:最大分组数
    :return: 分割点
    """
    fea_count = df[[fea_name, target_name]].copy().groupby([fea_name, target_name]).size().unstack().fillna(0.0)
    while fea_count.shape[0] > max_bin_count:
        chi_list = []
        for i in range(fea_count.shape[0] - 1):
            chi_value = chi2(fea_count.iloc[i:i + 2].values)
            chi_list.append([fea_count.index[i], chi_value])
        chi_min_index = np.argmin(np.array(chi_list)[:, 1])
        if chi_min_index == len(chi_list) - 1:
            current_fea = chi_list[chi_min_index][0]
            fea_count.loc[current_fea] = fea_count.loc[current_fea:].sum(axis=0)
            fea_count = fea_count.loc[:current_fea].copy()
        else:
            current_fea = chi_list[chi_min_index][0]
            next_fea = chi_list[chi_min_index + 1][0]
            fea_count.loc[current_fea] = fea_count.loc[current_fea] + fea_count.loc[next_fea]
            fea_count.drop([next_fea], inplace=True)
            chi_list.remove(chi_list[chi_min_index + 1])
    return fea_count


def discrete(path):
    df = pd.read_csv(path)
    target_name = df.columns[-1]
    fea_names = df.columns[0:-1]
    dis_count = 2
    for f in fea_names:
        chi_merge(df, f, target_name, dis_count)


if __name__ == '__main__':
    discrete('iris.csv')
