import pandas as pd
import numpy as np
from model.get_data_path import get_train_data_path

def get_new_mergeddata(filename):
    df = pd.read_csv(filename, encoding='utf-8', index_col=0)
    data_set = df.iloc[:, :].values
    x_train = data_set[:,:-1]
    y_train = data_set[:,-1]
    # tmp_data_set=[]
    # for i in range(len(data_set)):
    #     if i % 2==0:
    #         tmp_data_set.append(data_set[i])
    # print(np.shape(np.array(tmp_data_set)))
    tmp_x_batch = []
    tmp_y_batch = []
    print(x_train[:3])
    print(y_train[:3])
filename=get_train_data_path()
get_new_mergeddata(filename)