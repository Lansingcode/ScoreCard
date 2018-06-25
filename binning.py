# -*- coding:utf-8 -*-
__author__ = 'xujia'

import pandas as pd
import numpy as np
from scipy import stats


def equal_distance_binning(df, fea_name, bin_count):
    """
    等距分箱
    :param df:
    :param fea_name:
    :param bin_count
    :return:
    """
    df[fea_name + '_d'] = pd.cut(df[fea_name], bin_count)


def equal_frequency_binning(df, fea_name, bin_count):
    """
    等频分箱
    :param df:
    :param fea_name:
    :param bin_count
    :return:
    """
    df[fea_name + '_f'] = pd.cut(df[fea_name], bin_count)


def auto_binning(df, target_name, feature_name, max_bin_count):
    """
    自动分箱
    :param df:
    :param target_name: 目标变量名
    :param feature_name:特征变量名称
    :param max_bin_count:最大分箱数
    :return:
    """
    r = 0
    good = df[target_name].sum()
    bad = df[target_name].count() - good
    while np.abs(r) < 1:
        d1 = pd.DataFrame({'X': df[feature_name],
                           'Y': df[target_name],
                           'Bucket': pd.qcut(df[feature_name], max_bin_count, duplicates='drop')})
        d2 = d1.groupby('Bucket', as_index=True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        max_bin_count = max_bin_count - 1
    woe = np.log((d2.mean().Y / (1 - d2.mean().Y)) / (good / bad))
    woe_dict = woe.to_dict()
    woe_values = sorted(list(woe_dict.values()))
    print(woe_values)
    # 如果存在woe为inf情况，将其替换为不为inf的最大值加一
    df[feature_name + '_woe'] = d1['Bucket'].apply(lambda x: woe_dict[x]) \
        .replace(np.inf, woe_values[-2] + 1) \
        .replace(-np.inf, woe_values[1] - 1)


def chi2(A):
    ''' Compute the Chi-Square value '''
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


def chi_merge(df, fea_name, target_name, dis_count):
    fea_count = df[[fea_name, target_name]].copy().groupby([fea_name, target_name]).size().unstack().fillna(0.0)
    while fea_count.shape[0] > dis_count:
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
    print(fea_count)


def discrete(path):
    df = pd.read_csv(path)
    target_name = df.columns[-1]
    fea_names = df.columns[0:-1]
    dis_count = 2
    for f in fea_names:
        chi_merge(df, f, target_name, dis_count)


if __name__ == '__main__':
    discrete('iris.csv')
