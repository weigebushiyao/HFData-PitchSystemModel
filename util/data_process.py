#-*-coding:utf-8-*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# x=np.array(range(14))
# y=np.array(range(140))
# plt.plot(x,y,c='r')
# plt.show()
import pandas as pd


# utf编码格式的csv文件中的中文一般会是乱码，这时需要把文件格式另存为gbk格式
def csv_utf_2_gbk(srcPath):
    try:
        data = pd.DataFrame(pd.read_csv(srcPath, encoding='utf8', low_memory=False))
        data.to_csv(srcPath, index=False, sep=',', encoding='gbk')
    except:
        print(srcPath, "文件处理出错")


def csv_gbk_2_utf(srcPath):
    try:
        data = pd.DataFrame(pd.read_csv(srcPath, encoding='gbk', low_memory=False))
        data.to_csv(srcPath, index=False, sep=',', encoding='utf8')
    except:
        print(srcPath, "文件处理出错")

# csv_gbk_2_utf('../data/2018_12/rzjx_UP2.0_201801w002.csv')
# csv_gbk_2_utf('../data/2018_12/rzjx_UP2.0_201801w002.csv')
def csv_2_xlsx(srcPath):
    try:
        data = pd.DataFrame(pd.read_csv(srcPath, encoding='utf8', low_memory=False))
    except:
        data = pd.DataFrame(pd.read_csv(srcPath, encoding='gbk', low_memory=False))
    data.to_excel(srcPath[:-3] + 'xlsx', sheet_name='Sheet1', index=False)

#csv_2_xlsx('../data/2018_12/rzjx_UP2.0_201801w002.csv')

# df=pd.read_csv('../data/originalData/2018_12/rzjx_UP2.0_201804w001.csv', encoding='utf-8', )
# print(df)
# #
# # print(df.shape)
# # #df.loc[:, ~df.columns.str.contains('^Unnamed')]
# new_df=df.drop(labels=range(20,23),inplace=False,axis=0)
# print(new_df)
# #
# new_df.to_csv('../data/2018_12/new_rzjx_UP2.0_201804w001.csv',index=False)
# #
# # s=df['锘�fjtime']
# # print(s)
# # #print(df.dtypes)
# t_name=df.columns.values[0]
# print(t_name)
# df.rename(columns={t_name:'fjtime'},inplace=True)
# #print(df.columns.values)
# print(df['fjtime'])
# #r=df.loc['w001|2018-02-0100:00:00':'w001|2018-02-0100:00:30']
# def processTime(t):
#     t=t
#     new_t=str.split(t,'|')[1]
#     new_t=new_t[:10]+' '+new_t[10:]
#     return new_t
# df['fjtime']=df.apply(lambda x:processTime(x['fjtime']),axis=1)
#
# time_list=['2018-02-01 00:00:00','2018-02-02 00:00:00']
# tr=df[df['fjtime'].between('2018-04-01 00:05:30','2018-04-01 0:05:40')]
#
# print(tr.index.values)




#df.to_csv('../data/2018_12/rzjx_UP2.0_201802w001.csv')
# # print('是')
# # print(df.columns.values[0].decode(''))
#
# t=df.get(df.columns.values[0])
# print(t.shape)
# a=df.get('CI_WindSpeed1')
# b=df.get('CI_GearboxInputShaftTemp')
# a=pd.DataFrame(a)
# b=pd.DataFrame(b)
# print(a.shape)
# c=a.join(b)
# #c=t.join(c)
# # print(c.shape)
# # d=c.drop_duplicates(subset=('CI_WindSpeed1','CI_GearboxInputShaftTemp'),inplace=False)
# # print(d.shape)
# d=c.loc[:]
import datetime
import time
# future_mouth_first = datetime.datetime(2018, 10,1, 23, 59, 59)
#             # 当月最后一天最后一秒时间
# this_month = future_mouth_first - datetime.timedelta(days=1)
# print(this_month)
#
# first_day = datetime.datetime(2018, 9, 1, 23, 59, 59)
# # 上个月最后一天
# up_last = first_day - datetime.timedelta(days=1)
# up_last=datetime.datetime(2018,9,1)
# up_last=time.strptime(str(up_last),"%Y-%m-%d %H:%M:%S")
# print(up_last)
# a=dict()
# a['b']=[]
# a['b'].append(1)
# c='cc'
# if c not in a:
#     a[c]=1
# print(a)

now_time=time.strftime('%Y%m%d%H')

print(now_time)
