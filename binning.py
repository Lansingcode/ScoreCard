# -*- coding:utf-8 -*-
__author__ = 'xujia'

import pandas as pd


def equal_distance_binning(df, fea_name):
    """
    等距分箱
    :param df:
    :param fea_name:
    :return:
    """
    df[fea_name + '_d'] = pd.cut(df[fea_name])


def equal_frequency_binning(df, fea_name):
    """
    等频分箱
    :param df:
    :param fea_name:
    :return:
    """
    df[fea_name + '_f'] = pd.cut(df[fea_name])


def auto_binning(df, fea_name):
    """
    自动分箱
    :param df:
    :param fea_name:
    :return:
    """
    df[fea_name + '_a'] = pd.cut(df[fea_name])
