import os

cur_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(cur_path, '..'))


def get_train_data_path():
    data_path = parent_path + '/data/mergedData_new/'
    datafile = os.listdir(data_path)[0]
    return data_path + datafile


def get_test_data_path():
    data_path = parent_path + '/data/mergedFaultData_new/'
    datafile = os.listdir(data_path)[0]
    return data_path + datafile