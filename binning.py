# -*- coding:utf-8 -*-
__author__ = 'xujia'

import pandas as pd
import numpy as np
from scipy import stats


class Bin:
    def __init__(self, df, target_name, bin_count):
        self.df = df
        self.target_name = target_name
        self.bin_count = bin_count

    def equal_distance_binning(self, fea_name):
        """
        等距分箱
        :param fea_name:
        :return:
        """

        self.df[fea_name + '_d'] = pd.cut(self.df[fea_name], self.bin_count)
        fea_count = self.df[[fea_name + '_d', self.target_name]].copy().groupby(
            [fea_name + '_d', self.target_name]).size().unstack().fillna(0.0)
        fea_count.index = fea_count.index.map(lambda x: x.left)
        fea_count.index.name = fea_name
        return fea_count

    def equal_frequency_binning(self, fea_name):
        """
        等频分箱
        :param fea_name:
        :return:
        """
        self.df[fea_name + '_f'] = pd.cut(self.df[fea_name], self.bin_count)
        fea_count = self.df[[fea_name + '_f', self.target_name]].copy().groupby(
            [fea_name + '_f', self.target_name]).size().unstack().fillna(0.0)
        fea_count.index = fea_count.index.map(lambda x: x.left)
        fea_count.index.name = fea_name
        return fea_count

    def auto_binning(self, fea_name):
        """
        自动分箱
        :param fea_name:特征变量名称
        :return:
        """
        r = 0
        while np.abs(r) < 1:
            d1 = pd.DataFrame({'X': self.df[fea_name],
                               'Y': self.df[self.target_name],
                               fea_name + '_d': pd.qcut(self.df[fea_name], self.bin_count, duplicates='drop')})
            d2 = d1.groupby(fea_name + '_d', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            self.bin_count = self.bin_count - 1

        fea_count = self.df[[fea_name + '_d', self.target_name]].copy().groupby(
            [fea_name + '_d', self.target_name]).size().unstack().fillna(0.0)
        fea_count.index = fea_count.index.map(lambda x: x.left)
        fea_count.index.name = fea_name
        return fea_count

    def chi2(self, A):
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

    def chi_merge(self, fea_name):
        """
        chiMerge的主算法
        :param fea_name:需要进行分段的特征名称
        :return: 分割点
        """
        fea_count = self.df[[fea_name, self.target_name]].copy().groupby(
            [fea_name, self.target_name]).size().unstack().fillna(0.0)
        while fea_count.shape[0] > self.bin_count:
            chi_list = []
            for i in range(fea_count.shape[0] - 1):
                chi_value = self.chi2(fea_count.iloc[i:i + 2].values)
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
        fea_count.index = np.append([-np.inf], fea_count.index.values[1:])
        fea_count['bin'] = pd.cut(np.append(fea_count.index.values, [np.inf]),
                                  bins=np.append(fea_count.index.values, [np.inf]))[1:].astype(str)
        fea_count.index.name = fea_name
        return fea_count

#
# def discrete(path):
#     df = pd.read_csv(path)
#     target_name = df.columns[-1]
#     fea_names = df.columns[0:-1]
#     dis_count = 2
#     for f in fea_names:
#         chi_merge(df, f, target_name, dis_count)
#
#
# if __name__ == '__main__':
#     discrete('iris.csv')
