# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

train=pd.read_csv("../data/mergedData/datamining/wfzc_A2_cap_temp_1.csv", encoding='utf-8')
print(train.shape)

#train=train.dropna()
# train_label = train['pitch_Atech_capacitor_temp_1'].values
# train = train.drop(columns=['pitch_Atech_capacitor_temp_1'])

train=train.iloc[:,:].values

#引入数据
#输入训练和测试的缺失值
#train.fillna(-999,inplace=True)
# X = train.drop(['CI_GearboxOilSumpTemp'],axis=1)
# #设定x
# y = train.CI_GearboxOilSumpTemp
X=train[:,:-1]

y=train[:,-1]
#设定y
from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7)
#将x，y的70%进行训练模型，30%进行检验模型
#categorical_features_indices = np.where(X.dtypes != np.float)[0]
#导入库
model=CatBoostRegressor(iterations=1000,depth=2,learning_rate=0.2,loss_function='RMSE')
#设置参数

model.fit(X_train,y_train,cat_features=None,eval_set=(X_validation, y_validation),plot=True)
y_pred =model.predict(X_validation)
mse=mean_squared_error(y_validation,y_pred)
print('aa')
print(mse)
#利用30%的x代入模型预测y值
# y_pred=np.array(y_pred)
# y_validation=np.array(y_validation)
# mse_test=np.sum((y_pred-y_validation)**2)/len(y_train)
# #均方误差
# rmse_test=mse_test ** 0.5
# #均方根误差
# mae_test=np.sum(np.absolute(y_pred-y_validation))/len(y_validation)
# #平均绝对误差
# print(mse_test)
# print(rmse_test)
# print(mae_test)
# print(y_pred)
#输出得到的值


#设定x轴
#mse=y_pred-y_validation
y_v=[]
#mse_v=[]
pred_v=[]
for i in range(len(y_validation)):
    if i %200==0:
        y_v.append(y_validation[i])
        #mse_v.append(mse[i])
        pred_v.append(y_pred[i])
x=np.array(range(len(y_v)))
#30%预测的y值与已知的y值的误差
plt.plot(x, y_v, color="green", label="training accuracy")
plt.plot(x, pred_v, color="red", label="preding accuracy")
#plt.show()
#plt.plot(x,mse_v,color="blue")
#画图
plt.show()