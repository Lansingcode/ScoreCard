# -*-coding:utf-8 -*-

from sklearn import metrics
import matplotlib.pyplot as plt


def auc(model, test_data):
    """

    :param model:
    :param test_data:
    :param fea_list:
    :param target:
    :return:
    """
    predict_value = model.predict_proba(test_data.ix[:,0:-1])[:, 1]
    return metrics.roc_auc_score(test_data.ix[:,-1], predict_value)


def roc(model, test_data):
    """

    :param model:
    :param test_data:
    :param fea_list:
    :param target:
    :return:
    """
    predict_value = model.predict_proba(test_data.ix[:,0:-1])[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(test_data.ix[:,-1], predict_value)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='data1, AUC = %0.2f' % roc_auc)
    plt.legend(loc=4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Diagram")
    plt.show()
