import logging
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# 提供加载数据的接口
folderpath = os.path.abspath(os.path.dirname(__file__))
datapath = os.path.abspath(os.path.join(folderpath, '..')) + '/data/trainTestData/'


# 加载数据，并对缺失值比例进行判断，若超过50%则将该特征删除。若缺失值比例在设定范围内，则将该行删除，返回新的df。
def load_data(filename,null_rat=0.5):
    try:
        # f = open(datapath + filename, 'r')
        df = pd.read_csv(filename, encoding='utf-8',header=None)
        print('data shape:'+str(df.shape))
        # null_ratio=df.isnull().sum()/len(df)
        # if null_ratio>null_rat:
        #     return None,None
        #df=df.dropna()
        dataset = df.iloc[:100, :].values
        data_x=dataset[:,:-1]
        data_y=dataset[:,-1]
        #data_y=np.array(data_y).reshape((-1,1))
        return data_x, data_y
    except:
        logging.error("open data file failed")
        return None, None


# load_data(datapath)
# 划分训练集和测试集，train_size为训练样本的比例。
def train_test_dataset(filename):
    data_x, data_y = load_data(filename)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=0.8)  # list
    return x_train, y_train, x_test, y_test


# train_test_dataset(datapath)
# 交叉验证接口，返回k折验证的元数据索引以及数据集，可以用来直接生产k个训练样本和测试样本。
def cross_validation(filename, k):
    data_x, data_y = load_data(filename)
    data_x, data_y = np.array(data_x), np.array(data_y)
    train_idx_list, test_idx_list = [], []
    try:
        skf = StratifiedKFold(n_splits=k)
        for train_idx, test_idx in skf.split(data_x, data_y):
            # x_train,y_train=data_x[train_idx],data_y[train_idx]
            # x_test,y_test=data_x[test_idx],data_y[test_idx]
            train_idx_list.append(train_idx)
            test_idx_list.append(test_idx)
        return train_idx_list, test_idx_list, data_x, data_y
    except:
        logging.error("No dataset")
        return None, None, None, None
