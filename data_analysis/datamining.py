#-*-coding=utf-8-*-
import pandas as pd
import os
import sys
from util.featureselector import FeatureSelector
import time

"""
Feature importances will change on multiple runs of the machine learning model
Decide whether or not to keep the extra features created from one-hot encoding
Try out several different values for the various parameters to decide which ones work best for a machine learning task
The output of missing, single unique, and collinear will stay the same for the identical parameters
Feature selection is a critical step of a machine learning workflow that may require several iterations to optimize
"""
now_time=time.strftime('%Y%m%d')
hour_minute=time.strftime("%H%M")
cur_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(cur_path, '..'))
data_path = parent_path + '/data/mergedData_new/'
if not os.path.exists(data_path):
    sys.exit(1)
result_path=cur_path+'/result/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
single_result_path=result_path+'result_'+hour_minute+'/'
if not os.path.exists(single_result_path):
    os.makedirs(single_result_path)
class DataMining():
    def __init__(self):
        #self.dataFile = data_path + os.listdir(data_path)[0]
        self.dataFile=data_path+'datamining/wfzc_A2_cap_temp_1.csv'
        self.figure_path=single_result_path

    def dataAnalysis(self):
        df = pd.read_csv(self.dataFile, encoding='utf-8',index_col=0)#省去索引
        des=df.describe()
        des.to_csv(single_result_path+'data_analysis_result.csv',encoding='utf-8')
        #print(des)
        print(df.columns.values)
        df.dropna()
        #df = df[9990:200000]
        print(df.shape)
        train_label = df['pitch_Atech_capacitor_temp_1'].values
        df = df.drop(columns=['pitch_Atech_capacitor_temp_1'])
        fs = FeatureSelector(data=df, labels=train_label)
        fs.figure_path=self.figure_path
        self.missingValueAnalysis(fs)
        self.singleValueAnalysis(fs)
        self.collinearFeatureAnalysis(fs,thr=0.9)
        fs.identify_zero_importance(task='regression',eval_metric='L2',n_iterations=10,early_stopping=True)
        one_hot_features=fs.one_hot_features
        base_features=fs.base_features
        print("There are %d original features" % len(base_features))
        print('There are %d one-hot feature' % len(one_hot_features))
        fs.data_all.head()
        zeroimportancefeature=fs.ops['zero_importance']
        print(zeroimportancefeature)

        fs.plot_feature_importances(threshold=0.9,plot_n=10)
        fs.feature_importances.head(9)

        fs.identify_low_importance(cumulative_importance=0.9)
        lowimportancefeatures=fs.ops['low_importance']
        lowimportancefeatures[:5]
        #
        # removemissingvalue=fs.remove(methods=['missing'])
        # removezeroimportance=fs.remove(methods=['missing','zero_importance'])
        # alltoremoved=fs.check_removal()
        # print(alltoremoved)
        #简便的进行特征筛选
        # dataremoved=fs.remove(methods='all')
        # dataremovedall=fs.remove(methods='all',keep_one_hot=False)
        # fs = FeatureSelector(data=df, labels=train_label)
        #
        # fs.identify_all(selection_params={'missing_threshold': 0.6, 'correlation_threshold': 0.98,
        #                                   'task': 'classification', 'eval_metric': 'L2',
        #                                   'cumulative_importance': 0.99})



   # def lowImportanceFeatures(self):

    def zeroImportanceFeature(self,fs,):
        fs.identify_zero_importance(task='regresssion', eval_metric='auc', n_iterations=20, early_stopping=True)
        one_hot_features = fs.one_hot_features
        base_features = fs.base_features
        print("There are %d original features" % len(base_features))
        print('There are %d one-hot feature' % len(one_hot_features))
        fs.data_all.head()
        zeroimportancefeature = fs.ops['zero_importance']
        print(zeroimportancefeature)

    def collinearFeatureAnalysis(self,fs,thr):
        fs.identify_collinear(correlation_threshold=thr)
        correlated_features = fs.ops['collinear']
        print(correlated_features)
        fs.plot_collinear()
        fs.plot_collinear(plot_all=True)
        fs.record_collinear.head()

    def singleValueAnalysis(self,fs):
        fs.identify_single_unique()
        single_unique = fs.ops['single_unique']
        print(single_unique)
        fs.plot_unique()
        fs.unique_stats.sample(5)

    def missingValueAnalysis(self,fs):
        fs.identify_missing(missing_threshold=0.6)
        missing_features = fs.ops['missing']
        print(missing_features)
        fs.plot_missing()
        fs.missing_stats.head(10)



dm = DataMining()
dm.dataAnalysis()
