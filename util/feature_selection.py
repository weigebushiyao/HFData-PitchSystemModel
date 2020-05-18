from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from util.data_load import train_test_dataset
import numpy as np
import os
import multiprocessing

n_cpu=multiprocessing.cpu_count()
print(n_cpu)
cur_path=os.path.abspath(os.path.dirname(__file__))
parent_path=os.path.abspath(os.path.join(cur_path,'..'))
result_path=parent_path+'/result'
datapath=parent_path+'/data/mergedData/'
filename=os.listdir(datapath)[0]
print(datapath+filename)
if not os.path.exists(result_path):
    os.makedirs(result_path)

train_x,train_y,test_x,test_y=train_test_dataset(datapath+filename)
print(test_y)

#计算模型的预测值准确率
def cal_Accuracy(pred,test_y):
    count=0
    pred_cnt=len(pred)
    for i in range(pred_cnt):
        if pred[i]==test_y[i]:
            count+=1
    return round(count/pred_cnt,2)

#使用随机森林对特征进行选择,并传递重要性阈值
def rank_feature_importance(result_name,thres):
    trees=250
    max_feat=8
    max_depth=30
    min_sample=2
    rfr=RandomForestRegressor(n_estimators=trees,max_features=max_feat,max_depth=max_depth,min_samples_split=min_sample,random_state=0,n_jobs=n_cpu*2+1)
    import time
    start_time=time.time()
    rfr.fit(train_x,train_y)
    end_time=time.time()
    first_time=round(end_time-start_time,2)
    data_shape=np.shape(np.array(train_x))
    pre=rfr.predict(test_x)
    acc=cal_Accuracy(pre,test_y)
    sfm=SelectFromModel(rfr,threshold=thres)
    sfm.fit(train_x,train_y)
    new_train_x=sfm.transform(train_x)
    new_test_x=sfm.transform(test_x)
    print(rfr.feature_importances_)
    print(data_shape)
    print(first_time)
    from util.show_save_result import ShowAndSave
    sas=ShowAndSave(pred=pre,true=test_y)
    sas.show_save_figure(modelname='catboost')
    new_data_shape=np.shape(new_train_x)
    start_time2=time.time()
    rfr.fit(new_train_x,train_y)
    end_time2=time.time()
    pre2=rfr.predict(new_test_x)
    acc2=cal_Accuracy(pre2,test_y)
    second_time=round(end_time2-start_time2,2)
    print(new_data_shape)
    print(second_time)
    print(acc2)
    result_txt={'original_data_shape':data_shape,'1st_train_time':first_time,'1st_train_accuracy':acc,'selected_data_shape':new_data_shape,'2nd_train_time':second_time,'2nd_train_accuracy':acc2}
    f=open(result_path+'/'+result_name,'w',newline='')
    for k,v in result_txt.items():
        f.write(k+': '+str(v)+'\n')
    f.close()

rank_feature_importance('fs_test',0.08)
#基于递归特征选择方法
def select_feature_RFE(n):
    model=LogisticRegression()
    rfe=RFE(model,n_features_to_select=n)
    rfe.fit(train_x,train_y)
    print(rfe.n_features_)
    print(rfe.support_)
    print(rfe.ranking_)
#select_feature_RFE(12)

def select_feature_ExtratTree():
    model=ExtraTreesRegressor()
    model.fit(train_x,train_y)
    print(model.feature_importances_)
#select_feature_ExtratTree()

