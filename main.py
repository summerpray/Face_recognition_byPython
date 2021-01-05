'''
可能会用到的函数
np.hstack((a,b)) a,b列合并
np.vstack((a,b)) a,b行合并
np.row_stack((a,b))    增加一行
np.column_stack((a,b)) 增加一列

#本项目使用PCA进行图像识别
'''
import numpy as np
from numpy import random
#np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from pylab import *
from PIL import Image
import os

rd = np.random.RandomState(888)
allFileNum = 0
matrix_all = [ ]
matrix_average = [ ]
#os.listdir(path)

#读入训练数据并且初始化
def read_img(path):
  for img_file in range(1,51):
    # 以下代码根据需要更改
    img = Image.open(path + '/' + str(img_file) + '.pgm')  # 读取文件
    img = array(img.convert('L'), 'f')
    arr = img.flatten()
    arr = list(map(int, arr))
    mat = np.transpose(np.mat(arr))
    if (img_file == 1):
      matrix_all = mat
      matrix_average = mat
    else:
      matrix_all = np.hstack((matrix_all,mat))
  return matrix_all

def PCA(X):
  #每个维度去中心化 如果是拉成列那么行去中心化 反之则反之
  np.save('X', X)
  trainNumber, perTotal = X.shape
  #mean(0)表示列平均值 mean(1)表示行平均值
  meanMatrix = X.mean(1)
  np.save('meanMatrix', meanMatrix)
  #去中心化的数据集其实就是平均脸
  X = X - meanMatrix
  X_T = np.transpose(X)
  #C是原始协方差矩阵
  C = np.mat(((1 / X.shape[0]) * X * X_T))
  #print(C,C.shape)
  # 求出特征值eigenvalue，特征向量featurevector
  #eigenvalue, featurevector = np.linalg.eig(C)
  #print(featurevector.shape)
  # 定义一个临时矩阵用来存储前K个特征向量
  #mat_temp = np.mat([])
  #mat_temp = featurevector[0]
  #for i in range(1, 50):
  #  mat_temp = np.vstack((mat_temp, featurevector[i]))
  #mat_temp = np.array(mat_temp)
  #np.save('feature_vector', mat_temp)
  feature_space = Read_mat()
  #生成特征脸
  #for i in range(0,50):
  #res = feature_space[0].reshape((92, 112))
  #res = Image.fromarray(res)
  #res = res.convert('L')
  #res.save('outfile.png')
  #特征脸就是平均脸在K组正交基对应的特征空间上的投影
  feature_face_all = feature_space * X
  np.save('feature_face_all', np.array(feature_face_all))
  #print(feature_face_all,feature_face_all.shape)
  print('OK!')

def Read_mat():
  a = np.load('feature_vector_noshape.npy')
  return a

def figure_rec():
  img = Image.open(path + '/' + str(img_file) + '.pgm')

if __name__ == "__main__":

  c = read_img("F:/facialRec/test/test")

  PCA(c)

