# -*- coding:utf-8 -*-
__author__ = 'xujia'

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel


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
    clf = DecisionTreeClassifier()
    clf = clf.fit(X, y)
    print(clf.feature_importances_)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print(X_new)


from minepy import MINE
m = MINE()