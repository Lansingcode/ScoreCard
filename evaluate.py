# -*-coding:utf-8 -*-

from sklearn import metrics
import matplotlib.pyplot as plt


def auc(model, test_data):
    """

    :param model:模型
    :param test_data:测试数据，dataframe格式，第一列至倒数第二列为特征字段，最后一列为目标字段
    :return:auc值
    """
    predict_value = model.predict_proba(test_data.ix[:, 0:-1])[:, 1]
    return metrics.roc_auc_score(test_data.ix[:, -1], predict_value)


def roc(model, test_data):
    """

    :param model:模型
    :param test_data:测试数据，dataframe格式，第一列至倒数第二列为特征字段，最后一列为目标字段
    :return:roc曲线
    """
    predict_value = model.predict_proba(test_data.ix[:, 0:-1])[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(test_data.ix[:, -1], predict_value)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='data1, AUC = %0.2f' % roc_auc)
    plt.legend(loc=4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Diagram")
    plt.show()


def correlation_coef(data):
    """
    计算相关系数
    :param data:
    :return:
    """
    correlation = data.corr()
    print(correlation)
    return correlation
