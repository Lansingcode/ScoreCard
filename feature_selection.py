# -*- coding:utf-8 -*-
__author__ = 'xujia'

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from minepy import MINE

from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def chi2_select(X, y, number):
    """
    根据卡方筛选变量，
    :param X:
    :param y:
    :param number:
    :return:
    """
    X_new = SelectKBest(chi2, k=number).fit(X, y)
    print(X_new.scores_)
    return X_new


def fea_select(X, y):
    """
    使用决策树筛选变量
    :param X:
    :param y:
    :return:
    """
    clf = DecisionTreeClassifier()
    clf = clf.fit(X, y)
    print(clf.feature_importances_)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print(X_new)
    return X_new


def mi(X, y):
    """
    计算互信息
    :param X:
    :param y:
    :return:
    """
    mi_dict = {}
    m = MINE()
    try:
        if X.shape[1] > 1:
            for f in X.columns:
                m.compute_score(X[f], y)
                mi_dict[f] = m.mic()
            print(mi_dict)
            return mi_dict
    except:
        m.compute_score(X, y)
        mi_dict[X.name] = m.mic()
        print(mi_dict)
        return mi_dict
