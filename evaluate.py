from sklearn  import metrics
import matplotlib.pyplot as plt


def auc(model, test_data, fea_list, target):
    predict_value = model.predict_proba(test_data[fea_list])[:, 1]
    return metrics.roc_auc_score(test_data[target], predict_value)


def roc(model, test_data, fea_list, target):
    predict_value = model.predict_proba(test_data[fea_list])[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(test_data[target], predict_value)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='data1, AUC = %0.2f' % roc_auc)
    plt.legend(loc=4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Diagram")
    plt.show()

