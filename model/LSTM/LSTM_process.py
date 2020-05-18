import tensorflow as tf

import numpy as np
import pandas as pd
from model.get_data_path import get_train_data_path,get_test_data_path
from sklearn.model_selection import train_test_split

class LSTM_Model:
    def __init__(self):
        self.time_step=5
        self.rnn_units=32
        self.batch_size=60
        self.input_size=6
        self.output_size=1
        self.lr=0.001
        self.weights=None
        self.bias=None
        self.x_placeholder=None
        self.y_placeholder=None

    def get_data(self):
        train_data=get_train_data_path()
        df=pd.read_csv(train_data,encoding='utf-8',index_col=0)
        data_set=df.iloc[:,:].values
        x_train=data_set[:,:-1]
        y_train=data_set[:,-1]
        normalized_data=(x_train-np.mean(x_train))/np.std(x_train)
        tmp_x_batch=[]
        tmp_y_batch=[]
        for i in range(len(data_set)-self.time_step-1):
            x=normalized_data[i:i+self.time_step]
            y=y_train[i+self.time_step-1]
            tmp_x_batch.append(x)
            tmp_y_batch.append(y)
        return tmp_x_batch,tmp_y_batch

    def init_params(self):

        self.x_placeholder=tf.placeholder(tf.float32,[None,self.time_step,self.input_size])
        self.y_placeholder=tf.placeholder(tf.float32,[None,self.time_step,self.output_size])
        self.weights={
            'in':tf.Variable(tf.random_normal_initializer([self.input_size, self.rnn_units])),
            'out':tf.Variable(tf.random_normal_initializer([self.rnn_units,self.output_size]))
        }
        self.bias={
            'in':tf.Variable(tf.constant(0.1,shape=[1,self.rnn_units])),
            'out':tf.Variable(tf.constant(0.1,shape=[1,self.output_size]))
        }

    def train_lstm(self):
        self.init_params()
        w_in=self.weights['in']
        b_in=self.bias['in']
        input=tf.reshape(self.x_placeholder,[-1,self.input_size])
        input=tf.matmul(input,w_in)+b_in
        input=tf.nn.sigmoid(input)
        cell=tf.nn.rnn_cell.BasicLSTMCell(self.rnn_units)



lstm=LSTM_Model()
lstm.get_data()
lstm.train_lstm()
print(tf.__version__)