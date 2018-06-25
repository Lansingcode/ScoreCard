# -*- coding:utf-8 -*-
__author__ = 'xujia'

import pandas as pd
import numpy as np
import binning
import evaluate
import modeling


def file_info(file_path):
    """
    获取文件信息
    :param file_path: 文件路径
    :return: {字段名称：[字段类型，数据量，空值个数]}
    """
    info_dict = {}
    raw_data = pd.read_csv(file_path)
    for c in raw_data.columns:
        c_type = raw_data[c].dtype
        nc = raw_data[c].size - raw_data[c].notnull().sum()
        info_dict[c] = [c_type, raw_data[c].size, nc]  # 字段类型，数据量，空值个数
    return info_dict, raw_data


def change_type(df, fea_type_dict):
    """
    改变数据类型
    :param df:
    :param fea_type_dict:
    :return:
    """
    type_dict = {1: 'float64', 2: 'int64', 3: 'str'}
    feature_dict = dict(zip(list(range(df.shape[1])), df.columns.values))

    print('当前数据类型为：')
    for (k, v) in fea_type_dict.items():
        print(k.rjust(15), v[0])

    print('字段名称对应数字为：')
    for (n, m) in feature_dict.items():
        print(n, m)
    fea_name = int(input('请输入如需要更改数据类型的字段对应的数字：'))
    if fea_name not in feature_dict.keys():
        fea_name = int(input('输入字段名称错误，请重新输入：'))
        if fea_name not in fea_dict.keys():
            pass
    fea_name = feature_dict[fea_name]

    target_type = int(input('请输入目标类型对应的数字(1: 浮点型(float64)，2: 整型(int64)，3: 字符型(str)：'))
    if target_type not in type_dict.keys():
        target_type = int(input('请输入目标类型对应的数字(1: 浮点型(float64)，2: 整型(int64)，3: 字符型(str)：'))
        if target_type not in type_dict.keys():
            pass
    target_type = type_dict[target_type]
    df[fea_name] = df[fea_name].astype(target_type)


def split_data(data_to_split):
    """
    数据分割
    :param data_to_split:带分割数据
    :return: （数据集1，数据集2）
    """
    # ratio = float(input('请输入数据分割比例：'))
    ratio = 0.8
    data_count = data_to_split.shape[0]
    selected_count = int(data_count * ratio)
    if selected_count > 0:
        splited_data = np.split(data.sample(frac=1), [selected_count], axis=0)
    else:
        return 'Data is too less'
    return splited_data


if __name__ == '__main__':
    # path=input('Please input the file path: ')
    path = 'iris.csv'
    fea_dict, data = file_info(path)
    data = data.fillna(0.0)
    # change_type(data, fea_dict)
    # print(data.dtypes)

    # print(t[0].shape)
    # print(t[1].shape)
    binning.auto_binning(data, 'Label', 'SepalLength', 10)
    binning.auto_binning(data, 'Label', 'PetalLength', 10)
    binning.auto_binning(data, 'Label', 'PetalWidth', 10)
    train_data, test_data = split_data(data)
    model = modeling.model(train_data, ['SepalLength_woe', 'PetalLength_woe', 'PetalWidth_woe'], 'Label')
    predict_score = modeling.score_trans(test_data[['SepalLength_woe', 'PetalLength_woe', 'PetalWidth_woe']], model, 0.5, 100, 10)
    print(list(zip(test_data['Label'].values, predict_score)))

    auc = evaluate.auc(model, test_data, ['SepalLength_woe', 'PetalLength_woe', 'PetalWidth_woe'], 'Label')
    print("auc: " + str(auc))
    evaluate.roc(model, test_data, ['SepalLength_woe', 'PetalLength_woe', 'PetalWidth_woe'], 'Label')
