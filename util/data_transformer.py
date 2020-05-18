from sklearn import preprocessing

#min-max normalization 对列进行处理，转化到[0,1]
def minMaxScaler(x):
    scaler=preprocessing.MinMaxScaler()
    newX=scaler.fit_transform(x)
    return newX

#标准化，实现正态分布。针对列。
def standardScaler(x):
    
    scaler=preprocessing.StandardScaler()
    newX=scaler.fit_transform(x)
    return newX

#归一化，实现二项正则的效果。
def dataNormalize(x):
    normalizer=preprocessing.Normalizer()
    newX=normalizer.fit_transform(x)
    return newX