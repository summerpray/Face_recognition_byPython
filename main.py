'''
可能会用到的函数
np.hstack((a,b)) a,b列合并
np.vstack((a,b)) a,b行合并
np.row_stack((a,b))    增加一行
np.column_stack((a,b)) 增加一列

#本项目使用PCA进行图像识别

建立四个隐含层，用来识别左眼、右眼、鼻子、嘴巴
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
  X_T = np.transpose(X)
  #C是原始协方差矩阵
  C = (1 / X.shape[1]) * X * X_T
  #求出特征值eigenvalue，特征向量featurevector
  eigenvalue, featurevector = np.linalg.eig(C)
  #featurevector_T = np.transpose([featurevector])
  P = np.transpose(featurevector)
  print(P.shape)




if __name__ == "__main__":
  #初始化图片，转化为矩阵
  #a = img_init()
  #随机生成各激活值对应的权值
  #w = weight_init(a)
  #print(m)
  c = read_img("F:/facialRec/test/test")
  PCA(c)

