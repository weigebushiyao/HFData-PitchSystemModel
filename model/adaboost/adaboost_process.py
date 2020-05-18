import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from model.get_data_path import get_train_data_path,get_test_data_path
from sklearn.model_selection import train_test_split
import os
from util.show_save_result import ShowAndSave

cur_path = os.path.abspath(os.path.dirname(__file__))
datafile = get_train_data_path()


class AdaboostModel(ShowAndSave):
    def __init__(self, params=None, jobname='adbmodel'):
        super().__init__()
        self.job_name = jobname
        self.cur_path = cur_path
        self.init_param()
        self.params = params

    def adaboostmodel(self):
        df = pd.read_csv(datafile, encoding='utf-8', index_col=0)
        traindata = df.iloc[:, :].values
        x = traindata[:, :-1]
        y = traindata[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)  # list
        raw_model = AdaBoostRegressor(
            base_estimator=DecisionTreeRegressor(max_features=None, max_depth=None, min_samples_split=20,
                                                 min_samples_leaf=10, min_weight_fraction_leaf=0, max_leaf_nodes=None),
            learning_rate=0.1, loss='square', n_estimators=200)
        raw_model.fit(x_train, y_train)
        raw_model.save_model(self.model_path + self.job_name)
        pred = raw_model.predict(x_test)
        self.true = y_test
        self.pred = pred
        self.show_save_figure(fig_path=self.fig_path,modelname=self.job_name, detal_idx=500)
        t_mean = self.cal_mean(self.true)
        p_mean = self.cal_mean(self.pred)
        self.save_result(true_mean=t_mean, pred_mean=p_mean)

    def test_model(self):
        fault_test_file_path=get_test_data_path()
        df=pd.read_csv(fault_test_file_path,encoding='utf-8',index_col=0)
        data=df.iloc[:,:].values
        x=data[:10000,:-1]
        y=data[:,-1]
        raw_model=AdaBoostRegressor().load_model(self.model_file)
        pred=raw_model.predict(x)
        self.true=y
        self.pred=pred
        self.show_save_figure(fig_path=self.fault_data_test_figure_path,modelname=self.job_name,detal_idx=50)
        t_mean=self.cal_mean(self.true)
        p_mean=self.cal_mean(self.pred)
        self.save_result(self.fault_data_test_result_path,true_mean=t_mean,pred_mean=p_mean)


adb = AdaboostModel()
adb.adaboostmodel()
adb.test_model()