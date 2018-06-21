# encoding:utf-8
import pandas as pd
import numpy as np
import logging
# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')


class ARFilter(object):

    def __init__(self, threshold=0.05, dest_var='y'):
        self.threshold = threshold
        self.dest_var = dest_var
        logging.basicConfig()
        self.logger = logging.getLogger("default")
        self.logger.setLevel(level=logging.INFO)

    def info_value(self):
        """
        信息熵
        :return:
        """
        pass

    def chi_square(self):
        """
        卡方
        :return:
        """
        pass

    def train_cal_input(self, excel_name='input.csv'):
        """
        AR值筛选
        输入：宽表【变量1、变量2、目标变量】、筛选下限（默认0.05）、目标变量名称（默认y）
        输出：筛选后的变量列表【变量名称,AR值】（按照AR值降序排列）
        计算方式：使用单个变量与目标变量进行逻辑回归运算，返回模型的K-S值即为该变量的AR值。
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_curve
        data = pd.read_csv(excel_name)
        # 创建逻辑回归模型
        logit_model = LogisticRegression()
        final_list = []
        for col in data.columns.values[0:-1]:
            if col != self.dest_var:
                # 特征变量值
                X = data[col].values.reshape(-1, 1)
                # 拆分数据集为训练集与测试集
                x_train = X[:-20]
                x_test = X[-20:]
                # 目标变量值
                y = data[self.dest_var].values.reshape(-1, 1)
                y_train = y[:-20]
                y_test = y[-20:]
                # 数据拟合
                logit_model.fit(x_train, y_train)
                # 每一列与y列做预测
                # prob = logit_model.predict_proba(data[col].values.reshape(-1, 1))
                prob = logit_model.predict_proba(x_test)
                # prob[:, 1] 预测结果为两列，分别为0值可能性与1值可能性，此处取1值可能性
                # fpr, tpr, thresholds = roc_curve(data[self.dest_var].values.reshape(-1, 1), prob[:, 1])
                fpr, tpr, thresholds = roc_curve(y_test, prob[:, 1])
                from scipy import stats
                # AR = float(stats.ks_2samp(y_test, prob[:, 1].reshape(-1, 1)).statistic)
                # AR = float(stats.ks_2samp(y_test.ravel(), prob[:, 1]).statistic)
                # testDF = pd.DataFrame()
                # testDF['predict_proba'] = prob[:,1]
                # testDF['label'] = np.array(y_test)
                # print self.cal_ks(testDF)
                # print str(AR) + "-" * 30
                ks = abs(fpr - tpr).max()
                # print str(ks) + "*" * 30
                # print ks
                if ks > self.threshold:
                    final_list.append({'varName': col, "AR": ks})
                else:
                    self.logger.info('列：' + col + '的AR值为:' + str(ks) + ", 低于阈值：" + str(self.threshold))
        # AR值排序
        final_list.sort(key=lambda ar_dict: ar_dict['AR'], reverse=True)
        self.logger.info(pd.DataFrame(final_list))
        pd.DataFrame(final_list, columns=['varName', 'AR']).to_excel('result.xlsx', index=False)

    def cal_ks(self, data):
        """手动计算KS值"""
        #  对样本数据排序，根据预测值升序排序
        sorted_list = data.sort_values(['predict_proba'], ascending=True)
        total_good_count = sorted_list['label'].sum() * 1.0
        total_bad_count = (sorted_list.shape[0] - total_good_count) * 1.0
        max_ks = 0.0
        good_count = 0.0
        bad_count = 0.0
        for index, row in sorted_list.iterrows():
            if row['label'] == 0:
                bad_count += 1.0
            else:
                good_count += 1.0
            val = abs(bad_count/total_bad_count - good_count/total_good_count)
            max_ks = max(max_ks, val)
        return max_ks

    def cal_ar(self, excel_name='test.xlsx'):
        excel = pd.read_excel(excel_name)
        if excel.columns.size < 2:
            self.logger.error("未找到Excel数据源！")
            return
        dest_value = excel[self.dest_var]
        final_list = []
        # result_frame = pd.DataFrame(columns=['varName', 'AR'])
        for col in excel.columns:
            if col != self.dest_var:
                AR = float(stats.ks_2samp(excel[col], dest_value).statistic)
                final_list
        # self.logger.info(final_list)
        # final_list.append({'AR': 1.0, 'colName': u'var3'})
        # final_list.append({'AR': 0.8, 'colName': u'var4'})
        final_list.sort(key=lambda ar_dict: ar_dict['AR'], reverse=True)
        # self.logger.info("final result:" + str(final_list))
        # self.logger.info("123")
        self.logger.info(pd.DataFrame(final_list))
        pd.DataFrame(final_list, columns=['varName', 'AR']).to_excel('result.xlsx', index=False)

    def fill_empty_value(self, col_name, data, default_value=0):
        """
        缺失值填充
        输入：宽表【变量1、变量2、目标变量】，变量名称，缺失值填充值（默认0）
        计算方式：直接将指定变量中的缺失值用参数中的填充值进行填充
        输出：填充后的宽表，变量缺失率
        """
        # data = pd.read_excel(file_name)
        if col_name not in data.columns.values:
            self.logger.error("输入宽表中不存在指定变量")
            return
        else:
            empty_count = data[col_name].shape[0] - data[col_name].count()
            if empty_count > 0:
                self.logger.info('当前共' + str(data.shape[0]) + '个变量值，其中缺失值个数为' + str(empty_count))
                # 替换空串为NAN
                data[col_name] = data[col_name].replace(' ', np.nan).fillna(value=default_value)
                self.logger.info('填补后，缺失值个数为' + str(data[col_name].shape[0] - data[col_name].count()))
                # data.to_excel('result.xls', index=False)
                return data
            else:
                self.logger.info('当前不存在缺失值')

    def del_empty_value(self, data, empty_rate_threshold=0.5):
        """
        缺失值剔除
        输入：宽表【变量1、变量2、目标变量】，缺失率（默认0.5）
        计算方式：计算宽表中各个变量的缺失率，并剔除缺失率超过0.5的变量
        输出：处理后宽表
        """
        for col in data.columns.values:
            if col == 'y':
                continue
            empty_ratio = (data[col].shape[0] - data[col].count())/data[col].shape[0]
            if empty_ratio >= empty_rate_threshold:
                self.logger.info("变量：" + col + "缺失率为" + str(empty_ratio) + ",高于阈值：" + str(empty_rate_threshold))
                data = data.drop(col, axis=1)
        return data
        # data.to_excel(file_name.split(".")[0] + "_new." + file_name.split(".")[1], index=False)

    def console_input(self, prompt="", if_value=[], else_value=[], if_rtn="", else_rtn=""):
        rtn = input(prompt)
        if rtn.strip() in if_value:
            return if_rtn
        elif rtn.strip() in else_value or len(else_value) == 0:
            return else_rtn
        else:
            raise IOError("未匹配到条件")

    def file_info(self, path):
        """
        获取文件信息
        :param path: 文件路径
        :return: {字段名称：[字段类型，数据量，空值个数]}
        """
        info_dict = {}
        data = pd.read_csv(path)
        for c in data.columns:
            ctype = data[c].dtype
            nc = data[c].size - data[c].notnull().sum()
            info_dict[c] = [ctype, data[c].size, nc]  # 字段类型，数据量，空值个数
        return info_dict, data

    def is_contain_empty_value(self, file_dict):
        empty_col_list = []
        for item in file_dict:
            self.logger.info(file_dict[item])
            if int(file_dict[item][2]) > 0:
                self.logger.info("列" + item + "空值个数：" + str(file_dict[item][2]))
                empty_col_list.append(item)
        if len(empty_col_list) > 0:
            return True, empty_col_list
        else:
            return False, []

    def main(self):
        file_path = input("请输入待处理的文件名路径：")
        import os.path
        if os.path.isfile(file_path):
            file_dict, data = self.file_info(file_path)
            is_contain_empty_value, empty_col_list = self.is_contain_empty_value(file_dict)
            if is_contain_empty_value:
                self.logger.info("当前存在缺失值")
                is_fill_empty = self.console_input(prompt="是否需要填充数据？1：是，其他值：否", if_value=["1"], else_value=[],
                                                   if_rtn=True, else_rtn=False)
                if is_fill_empty:
                    for col in empty_col_list:
                        fill_value = input("请输入列" + col + "待填充的数据：")
                        self.logger.info("列" + col + "将填充数据：" + fill_value)
                        data = self.fill_empty_value(col_name=col, data=data, default_value=fill_value)
                    print(data)
                else:
                    self.logger.info("不填充数据，程序退出")
            else:
                self.logger.info("当前不存在缺失数据")
        else:
            self.logger.error("指定的文件路径不存在")


def run():
    ar = ARFilter()
    # ar.train_cal_input()
    # ar.fill_empty_value(col_name='emptyCol', file_name='empty.xls', default_value=0)
    # ar.del_empty_value(file_name="empty_ratio.xls")
    ar.main()


if __name__ == "__main__":

    run()

