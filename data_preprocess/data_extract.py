import pandas as pd
import os
import csv
from operator import itemgetter
import logging
#读取配置文件，获取数据表中所需字段
def get_var():
    try:
        f=open('./varience_name.csv')
        csv_file=pd.read_csv(f,header=None)
    except:
        logging.error("Open varience_name.csv failed.")
        return
    var_name=[]
    for e in csv_file[0]:
        var_name.append(e)
    return var_name
#get_var()

#获取文件夹，返回文件夹list
def get_data_folder():
    cur_path = os.path.abspath(os.path.dirname(__file__))
    up_path = os.path.abspath(os.path.join(cur_path, '..'))+'data'
    data_dir = up_path + '/originalData'
    train_data=up_path+'/trainTestData'
    data_folder = os.listdir(data_dir)
    folder_list=[]
    for df in data_folder:
        folder_list.append(data_dir+'/'+df)
    return folder_list,train_data

#根据文件夹获取文件名称
def get_data_file(datafolder):
    if datafolder:
        file_iter=os.walk(datafolder)
        file_list=[]
        for root,dirs,file in file_iter:
            pass
        for f in file:
            file_list.append(datafolder+'/'+f)
        return file_list
    else:
        logging.info("no datafolder")

#根据获取的文件名获取csv数据
def read_file(filename,var_name,savepath):
    try:
        savefile=open(savepath,'a+',newline='')
        writer=csv.writer(savefile)
    except:
        logging.error("Open savepath failed")
        savefile.close()

    try:
        f=open(filename,'r')
        csvfile=csv.DictReader(f,)
     # 这里用列名找到所有列的索引
        for row in csvfile:
            writer.writerow(list(itemgetter(*var_name)(row)))
    except:
        logging.error("open datafile failed")
        f.close()
    savefile.close()
    f.close()

def data_extract():
    #获取变量名
    var_name=get_var()
    #获取目录下所有的文件夹
    folder_list,train_data=get_data_folder()
    #根据文件夹名称获取数据文件
    file_list=[]
    if folder_list:
        for folder in folder_list:
            file_list+=get_data_file(folder)
    if var_name:
    #根据文件获取数据,且是指定的列
        for fl in file_list:
            read_file(fl,var_name,train_data+'/test.csv')


def merge_DistoryData():
    fileList=os.listdir()