import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd


class ShowAndSave:
    def __init__(self):
        self.fig_path = None
        self.pred = None
        self.true = None
        self.multi_model_path = None
        self.single_model_path = None
        self.result_path = None
        self.model_path = None
        self.job_name = None
        self.params_file_path=None
        self.cur_path = None
        self.params=None
        self.fault_data_test_result_path=None
        self.fault_data_test_figure_path=None


    def init_param(self):
        # parent_path=os.path.abspath(os.path.join(cur_path,'..'))
        now_time = time.strftime('%Y%m%d%H%M')
        hour_minute=time.strftime('%H%M')
        self.multi_model_path = self.cur_path + '/' + self.job_name
        if not os.path.exists(self.multi_model_path):
            os.makedirs(self.multi_model_path)
        self.single_model_path = self.multi_model_path + '/' + self.job_name + '_' + now_time + '/'
        self.model_path = self.single_model_path + 'model_'+hour_minute+'/'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.fig_path = self.single_model_path + 'figure_'+hour_minute+'/'
        if not os.path.exists(self.fig_path):
            os.makedirs(self.fig_path)
        self.result_path = self.single_model_path + 'result_'+hour_minute+'/'
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.fault_data_test_result_path=self.single_model_path+'test_result_'+hour_minute+'/'
        if not os.path.exists(self.fault_data_test_result_path):
            os.makedirs(self.fault_data_test_result_path)
        self.fault_data_test_figure_path=self.single_model_path+'test_figure_'+hour_minute+'/'
        if not os.path.exists(self.fault_data_test_figure_path):
            os.makedirs(self.fault_data_test_figure_path)
        self.params_file_path=self.single_model_path+'params_file_'+hour_minute+'/'
        if not os.path.exists(self.params_file_path):
            os.makedirs(self.params_file_path)

    def cal_error(self):
        # 均方误差
        mse = np.sum((self.pred - self.true) ** 2 / len(self.true))
        # 均方根误差
        rmse = mse ** 0.5
        # 平均绝对值误差
        mae = np.sum(np.absolute(self.pred - self.true)) / len(self.true)
        return mse, rmse, mae

    def show_save_figure(self, fig_path,modelname=None, detal_idx=10):
        true_li = []
        error_li = []
        pred_li = []
        error = self.true - self.pred
        for i in range(len(self.true)):
            if i % detal_idx == 0:
                true_li.append(self.true[i])
                error_li.append(error[i])
                pred_li.append(self.pred[i])
        x = np.array(range(len(true_li)))
        # 30%预测的y值与已知的y值的误差
        plt.plot(x, true_li, color="green", label="true")
        plt.plot(x, pred_li, color="red", label="pred")
        # plt.show()
        plt.plot(x, error_li, color="blue", label='error')  # 画图
        plt.legend(loc='upper left',bbox_to_anchor=(0.1,0.95))
        plt.title(self.job_name)
        plt.savefig(fig_path + modelname)
        plt.show()

    def save_result(self,save_path, true_mean=None, pred_mean=None):
        mse, rmse, mae = self.cal_error()
        mape=self.mean_absolute_percentage_error()
        rsqu=self.r_square()
        data = {'true_mean':true_mean,'pred_mean':pred_mean,'mse': mse, 'rmse': rmse, 'mae': mae,'mape':mape,'R Square':rsqu,'params':self.params}
        df = pd.DataFrame(list(data.items()))
        df.to_csv(save_path + 'result.csv',encoding='utf-8',index=None,header=None)

    def cal_mean(self, input):
        mean_val=np.mean(input)
        return mean_val

    #平均相对误差绝对值，用于刻画预测值和真实值之间的偏差，越小越好
    def mean_absolute_percentage_error(self):
        y_true, y_pred = np.array(self.true), np.array(self.pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    #用于刻画预测值与真实值之间的拟合程度，越大越好
    def r_square(self):
        #sse:预测数据与原始数据对应点的误差的平方和
        #ssr:预测数据与原始数据均值之差的平方和
        #sst：原始数据与均值之差的平方和

        true_mean=self.cal_mean(self.true)
        sse=0
        sst=0
        for i in range(len(self.pred)):
            sse+=(self.pred[i]-self.true[i])**2
            sst+=(self.true[i]-true_mean)**2
        r_square=1-sse/sst
        return r_square