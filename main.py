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

#os.listdir(path)

#读入训练数据并且初始化
def read_img(path):
  for img_file in range(1,51):
    print(img_file)  # 打印当前读取的图片名
    # 以下代码根据需要更改
    img = Image.open(path + '/' + str(img_file) + '.pgm')  # 读取文件
    img = array(img.convert('L'), 'f')
    arr = img.flatten()
    arr = list(map(int, arr))
    mat = np.mat(arr)
    if (img_file == 1):
      matrix_all = mat
    else:
      matrix_all = np.vstack((matrix_all,mat))
    print(matrix_all)


def showimg(img, isgray=False):
  plt.axis("off")
  if isgray == True:
    plt.imshow(img, cmap='gray')
  else:
    plt.imshow(img)
  plt.show()

if __name__ == "__main__":
  #初始化图片，转化为矩阵
  #a = img_init()
  #随机生成各激活值对应的权值
  #w = weight_init(a)
  #print(m)
  read_img("F:/facialRec/test/test")


