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
    df[feature_name + '_woe'] = d1['Bucket'].apply(lambda x: woe_dict[x])\
                                            .replace(np.inf, woe_values[-2] + 1)\
                                            .replace(-np.inf, woe_values[1] - 1)
    # return woe_dict
