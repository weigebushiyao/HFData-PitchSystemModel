import matplotlib.pyplot as plt
from util.data_load import load_data
from mpl_toolkits.mplot3d import Axes3D
import os
from util import data_decomposition
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(cur_path, '..'))
savepath = parent_path + '/images'
if not os.path.exists(savepath):
    os.makedirs(savepath)

x, y = load_data('a_test.csv')
x=np.array(x)
y=np.array(y)
z=np.hstack((x,y))


# 绘制数据直方图,即为数据频率分布图
def plot_histogram(z, img_name):
    x = data_decomposition.pca_decomposition(z)
    newX, newY = [], []
    for e in x:
        newX.append(round(e[0],2))
        newY.append(e[1])
    print(newX)
    plt.hist(newX)
    plt.title('Data Histogram')
    plt.show()
    plt.savefig(savepath + '/' + img_name)
plot_histogram(z,'test-histogram')

# 绘制数据点图3-d
def plot_3d_point( z, img_name):
    x = data_decomposition.pca_decomposition(z, 3)
    newX, newY, newZ = [], [], []
    for e in x:
        newX.append(e[0])
        newY.append(e[1])
        newZ.append(e[2])
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title(img_name)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(newX,newY,newZ, c='r', marker='.')
    plt.show()
    plt.savefig(savepath + '/' + img_name,bbox_inches='tight')
#plot_3d_point(newX,newY,newZ,'test-3d')

# 绘制数据点图2-d
def plot_2d_point(z, img_name):
    x = data_decomposition.pca_decomposition(z)
    newX, newY = [], []
    for e in x:
        newX.append(e[0])
        newY.append(e[1])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_title(img_name)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.scatter(newX,newY,c='r',marker='.')
    plt.show()
    plt.savefig(savepath + '/' + img_name)
#plot_2d_point(newX,newY,'test-2d')