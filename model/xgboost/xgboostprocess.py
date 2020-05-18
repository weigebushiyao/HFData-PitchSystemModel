#-*-coding:utf-8-*-
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from xgboost.sklearn import XGBRegressor
from model.get_data_path import get_train_data_path,get_test_data_path
from sklearn.model_selection import train_test_split
import os
from util.show_save_result import ShowAndSave


cur_path=os.path.abspath(os.path.dirname(__file__))
datafile = get_train_data_path()


class XgboostModel(ShowAndSave):
    def __init__(self, params=None,jobname='xgb_model'):
        super().__init__()
        self.job_name=jobname
        self.cur_path=cur_path
        self.init_param()
        self.params = params
        self.model_file=self.model_path + self.job_name

    def xgboostmodel(self):
        df = pd.read_csv(datafile, encoding='utf-8', index_col=0)
        print(df.shape)
        traindata = df.iloc[:, :].values
        x = traindata[:, :-1]
        y = traindata[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)  # list
        if self.params is None:
            params={'max_depth':80,'n_estimators':512}
        else:
            params=self.params
        raw_model = XGBRegressor(max_depth=128,n_estimators=768,learning_rate=0.01,silence=False)
        raw_model.fit(x_train, y_train)
        raw_model.save_model(self.model_file)
        pred = raw_model.predict(x_test)
        self.true=y_test
        self.pred=pred
        self.show_save_figure(fig_path=self.fig_path,modelname=self.job_name, detal_idx=500)
        t_mean=self.cal_mean(self.true)
        p_mean=self.cal_mean(self.pred)
        self.save_result(self.result_path,true_mean=t_mean, pred_mean=p_mean)

    def test_model(self,model_file=None):
        if model_file is None:
            modelfile=self.model_file
        else:
            modelfile=self.single_model_path+'model_'+str(model_file)
        fault_test_file_path=get_test_data_path()
        df=pd.read_csv(fault_test_file_path,encoding='utf-8',index_col=0)
        data=df.iloc[:,:].values
        x=data[:,:-1]
        y=data[:,-1]
        xgb=XGBRegressor()
        raw_model=xgb.load_model(modelfile)
        pred=raw_model.predict(x)
        self.true=y
        self.pred=pred
        self.show_save_figure(fig_path=self.fault_data_test_figure_path,modelname=self.job_name,detal_idx=10)
        t_mean=self.cal_mean(self.true)
        p_mean=self.cal_mean(self.pred)
        self.save_result(self.fault_data_test_result_path,true_mean=t_mean,pred_mean=p_mean)

    def params_tuned(self):
        xgb=XGBRegressor(objective='reg:squarederror')
        params={'max_depth':[90,100,128],'n_estimators':[768,800,850]}
        grid=RandomizedSearchCV(xgb,params,cv=3,scoring='neg_mean_squared_error',n_iter=6)
        df = pd.read_csv(datafile, encoding='utf-8', index_col=0)
        traindata = df.iloc[100000:700000, :].values
        x = traindata[:, :-1]
        y = traindata[:, -1]
        grid.fit(x,y)
        print(grid.best_score_)
        print(grid.best_params_)
        self.params=grid.best_params_
        df=pd.DataFrame(list(self.params.items()))
        df.to_csv(self.params_file_path+'params.csv',encoding='utf-8',index=None,header=None)


xgb = XgboostModel()
#xgb.params_tuned()
xgb.xgboostmodel()
#xgb.test_model()
