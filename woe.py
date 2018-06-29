# -*- coding:utf-8 -*-

import pandas as pd
from math import log
import numpy as np
import math
from scipy import stats
from sklearn.utils.multiclass import type_of_target


class WOE:
    def __init__(self):
        self._WOE_MIN = -20
        self._WOE_MAX = 20

    def woe(self, X, y, event=1):
        '''
        Calculate woe of each feature category and information value
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param y: 1-D numpy array target variable which should be binary
        :param event: value of binary stands for the event to predict
        :return: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
                 numpy array of information value of each feature
        '''
        self.check_target_binary(y)
        X1 = self.feature_discretion(X)

        res_woe = []
        res_iv = []
        for i in range(0, X1.shape[-1]):
            x = X1[:, i]
            woe_dict, iv1 = self.woe_single_x(x, y, event)
            res_woe.append(woe_dict)
            res_iv.append(iv1)
        return np.array(res_woe), np.array(res_iv)

    def woe_single_x(self, x, y, event=1):
        """
        calculate woe and information for a single feature
        :param x: 1-D numpy starnds for single feature
        :param y: 1-D numpy array target variable
        :param event: value of binary stands for the event to predict
        :return: dictionary contains woe values for categories of this feature information value of this feature
        """
        self.check_target_binary(y)

        event_total, non_event_total = self.count_binary(y, event=event)
        x_labels = np.unique(x)
        woe_dict = {}
        iv = 0
        for x1 in x_labels:
            y1 = y[np.where(x == x1)[0]]
            event_count, non_event_count = self.count_binary(y1, event=event)
            rate_event = 1.0 * event_count / event_total
            rate_non_event = 1.0 * non_event_count / non_event_total
            if rate_event == 0:
                woe1 = self._WOE_MIN
            elif rate_non_event == 0:
                woe1 = self._WOE_MAX
            else:
                woe1 = math.log(rate_event / rate_non_event)
            woe_dict[x1] = woe1
            iv += (rate_event - rate_non_event) * woe1
        return woe_dict, iv

    def woe_replace(self, X, woe_arr):
        """
        replace the explanatory feature categories with its woe value
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param woe_arr: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
        :return: the new numpy array in which woe values filled
        """
        if X.shape[-1] != woe_arr.shape[-1]:
            raise ValueError('WOE dict array length must be equal with features length')

        res = np.copy(X).astype(float)
        idx = 0
        for woe_dict in woe_arr:
            for k in woe_dict.keys():
                woe = woe_dict[k]
                res[:, idx][np.where(res[:, idx] == k)[0]] = woe * 1.0
            idx += 1
        return res

    def combined_iv(self, X, y, masks, event=1):
        """
        calcute the information value of combination features
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param y: 1-D numpy array target variable
        :param masks: 1-D numpy array of masks stands for which features are included in combination,
                      e.g. np.array([0,0,1,1,1,0,0,0,0,0,1]), the length should be same as features length
        :param event: value of binary stands for the event to predict
        :return: woe dictionary and information value of combined features
        """
        if masks.shape[-1] != X.shape[-1]:
            raise ValueError('Masks array length must be equal with features length')

        x = X[:, np.where(masks == 1)[0]]
        tmp = []
        for i in range(x.shape[0]):
            tmp.append(self.combine(x[i, :]))

        dumy = np.array(tmp)
        # dumy_labels = np.unique(dumy)
        woe, iv = self.woe_single_x(dumy, y, event)
        return woe, iv

    def combine(self, list):
        res = ''
        for item in list:
            res += str(item)
        return res

    def count_binary(self, a, event=1):
        event_count = (a == event).sum()
        non_event_count = a.shape[-1] - event_count
        return event_count, non_event_count

    def check_target_binary(self, y):
        """
        check if the target variable is binary, raise error if not.
        :param y:
        :return:
        """
        y_type = type_of_target(y)
        if y_type not in ['binary']:
            raise ValueError('Label type must be binary')

    def feature_discretion(self, X):
        """
        Discrete the continuous features of input data X, and keep other features unchanged.
        :param X : numpy array
        :return: the numpy array in which all continuous features are discreted
        """
        temp = []
        for i in range(0, X.shape[-1]):
            x = X[:, i]
            x_type = type_of_target(x)
            if x_type == 'continuous':
                x1 = self.discrete(x)
                temp.append(x1)
            else:
                temp.append(x)
        return np.array(temp).T

    def discrete(self, x):
        """
        Discrete the input 1-D numpy array using 5 equal percentiles
        :param x: 1-D numpy array
        :return: discreted 1-D numpy array
        """
        res = np.array([0] * x.shape[-1], dtype=int)
        for i in range(5):
            point1 = stats.scoreatpercentile(x, i * 20)
            point2 = stats.scoreatpercentile(x, (i + 1) * 20)
            x1 = x[np.where((x >= point1) & (x <= point2))]
            mask = np.in1d(x, x1)
            res[mask] = (i + 1)
        return res

    def woe_feature(self, x, dict):
        new_x = []
        for i in x:
            new_x.append(dict[i])
        return new_x

    @property
    def WOE_MIN(self):
        return self._WOE_MIN

    @WOE_MIN.setter
    def WOE_MIN(self, woe_min):
        self._WOE_MIN = woe_min

    @property
    def WOE_MAX(self):
        return self._WOE_MAX

    @WOE_MAX.setter
    def WOE_MAX(self, woe_max):
        self._WOE_MAX = woe_max


def add_woe_col(data, bins):
    """
    为指定特征添加一列对应的WOE值
    :param data:原始数据
    :param bins:分段信息
    :return:在原始数据上添加一列
    """
    fea_name = bins.index.name
    bin_index = bins.index.values.astype(float)
    bin_index[0] = -np.inf
    bins.index = bin_index
    bins.index.name = fea_name
    bin_index = np.append(bin_index, np.inf)
    interval_list = []
    woe_list = []
    max_woe = 10
    min_woe = -10
    for i in range(len(bin_index) - 1):
        if bin_index[i] == bin_index[i + 1]:
            continue
        else:
            interval_list.append('(' + str(bin_index[i]) + ', ' + str(bin_index[i + 1]) + ']')
            rate_event = bins[0.0][bin_index[i]] / bins[0.0].sum()
            rate_non_event = bins[1.0][bin_index[i]] / bins[1.0].sum()
            if rate_event == 0.0:
                woe_list.append(min_woe)
            elif rate_non_event == 0.0:
                woe_list.append(max_woe)
            else:
                woe_list.append(math.log(rate_event / rate_non_event))
    bin_woe = dict(zip(interval_list, woe_list))
    data[fea_name + '_bin'] = pd.cut(data[fea_name], bins=np.append(bins.index.values, [np.inf])).astype(str)
    data[fea_name + '_woe'] = data[fea_name + '_bin'].apply(lambda x: bin_woe[x])
    del data[fea_name + '_bin']

# if __name__ == '__main__':
#     path=input('Please input the file path: ')
#     path = 'iris.csv'
#     raw_data = pd.read_csv(path)
#     print(raw_data)
#     woe = WOE()
#     woe_result=woe.woe_single_x(x=raw_data,'SepalLength')
#     ret = pd.cut(raw_data['SepalLength'], 5)
#     print(ret)
