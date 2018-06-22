# -*- coding:utf-8 -*-
__author__ = 'xujia'
import numpy as np

from sklearn.linear_model import LogisticRegression


def model(data, fea_list, target):
    cls = LogisticRegression()
    cls.fit(data[fea_list], data[target])
    print(cls.coef_)
    print(cls.intercept_)
    return cls


def score_trans(data, coef, intercept, scaled_value, odds, pdo):
    a = (np.log(2 * odds) - np.log(odds)) / pdo
    b = np.log(odds, np.e) - a * scaled_value
    p = intercept + coef.dot(data)
    score = np.log(p / (1 - p)) * a + b
    return score
