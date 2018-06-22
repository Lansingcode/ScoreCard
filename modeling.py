# -*- coding:utf-8 -*-
__author__ = 'xujia'
import numpy as np

from sklearn.linear_model import LogisticRegression


def model(data, fea_list, target):
    cls = LogisticRegression()
    cls.fit(data[fea_list], data[target])
    return cls


def score_trans(data, model, p, scaled_value, pdo):
    b = pdo / np.log(2)
    a = scaled_value + b * np.log(p)
    p = model.predict_proba(data)[:, 1]
    score = a - np.log(p / (1 - p)) * b

    return score
