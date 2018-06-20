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

    def fill_empty_value(self, col_name, file_name='input.xls', default_value=0):
        """
        缺失值填充
        输入：宽表【变量1、变量2、目标变量】，变量名称，缺失值填充值（默认0）
        计算方式：直接将指定变量中的缺失值用参数中的填充值进行填充
        输出：填充后的宽表，变量缺失率
        """
        data = pd.read_excel(file_name)
        # print(str(np.nan))
        # print(type(str(np.nan)))
        # print type(str(data['emptyCol'][14]))
        # print len(str(data['emptyCol'][14]).strip())
        # print type(str(data['emptyCol'][14]).strip())
        if col_name not in data.columns.values:
            self.logger.error("输入宽表中不存在指定变量")
            return
        else:
            empty_count = data.shape[0] - data[col_name].count()
            if empty_count > 0:
                self.logger.info('当前共' + str(data.shape[0]) + '个变量值，其中缺失值个数为' + str(empty_count))
                # 替换空串为NAN
                # data[col_name] = data[col_name].replace(' ', np.nan).fillna(value=default_value)
                data['result'] = data[col_name].replace(' ', np.nan).fillna(value=default_value)
                # self.logger.info('填补后，缺失值个数为' + str(data.shape[0] - data[col_name].count()))
                self.logger.info('填补后，缺失值个数为' + str(data.shape[0] - data['result'].count()))
                data.to_excel('result.xls', index=False)
            else:
                self.logger.info('当前不存在缺失值')

    def del_empty_value(self, file_name='input.xls', empty_rate_threshold=0.5):
        """
        缺失值剔除
        输入：宽表【变量1、变量2、目标变量】，缺失率（默认0.5）
        计算方式：计算宽表中各个变量的缺失率，并剔除缺失率超过0.5的变量
        输出：处理后宽表
        """
        data = pd.read_excel(file_name)
        for col in data.columns.values:
            if col == 'y':
                continue
            empty_ratio = (data[col].shape[0] - data[col].count())/data[col].shape[0]
            if empty_ratio >= empty_rate_threshold:
                self.logger.info("变量：" + col + "缺失率为" + str(empty_ratio) + ",高于阈值：" + str(empty_rate_threshold))
                data = data.drop(col, axis=1)
        data.to_excel(file_name.split(".")[0] + "_new." + file_name.split(".")[1], index=False)


def run():
    ar = ARFilter()
    ar.train_cal_input()
    # ar.fill_empty_value(col_name='emptyCol', file_name='empty.xls', default_value=0)
    # ar.del_empty_value(file_name="empty_ratio.xls")

if __name__ == "__main__":
    run()
