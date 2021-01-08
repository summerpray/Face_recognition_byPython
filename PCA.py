'''
Author: summerpray

#本项目使用PCA进行图像识别
每个人脸训练7张 可以在下面更改
可能会用到的函数

np.hstack((a,b)) a,b列合并
np.vstack((a,b)) a,b行合并
np.row_stack((a,b))    增加一行
np.column_stack((a,b)) 增加一列
'''
import numpy as np
from numpy import random
#np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from PIL import Image
import os

allFileNum = 0
matrix_all = [ ]
matrix_average = [ ]
rd = np.random.RandomState(888)
# 保存特征脸的路径和数据集根目录
savepath = 'D:/Facial_Recognition/Face_recognition/feature_face/'
rootDir = 'D:/Facial_Recognition/Face_recognition/att_faces/'
Pic_Num = 7 # 每个数据集选取多少张图片
K = 50 # 取前K个特征向量

# 代码中有保存矩阵信息的语句可以根据是否需要更新矩阵来取消注释

# 从训练样本中获取数据
def get_train_img():
  # 定义一些中间变量
  mat = np.mat([])
  label_train = []
  label_rec = []
  mat_rec = np.mat([])
  mat_train = np.mat([])
  # 获取该目录下所有的文件名称和目录名称
  for lists in os.listdir(rootDir):
    listDir = rootDir + lists
    cnt = 0 # 用来计数
    for img_num in os.listdir(listDir):
      cnt += 1
      # 读取每个子文件夹下的图片
      arr = np.array([])
      img = Image.open(listDir + '/' + str(cnt) + '.pgm')  # 读取文件
      img = np.array(img.convert('L'), 'f')
      arr = img.flatten()
      mat = np.transpose(np.mat(arr))
      # 并且为每张脸存储一个对应的人脸标签
      if (cnt > Pic_Num):
        if (mat_rec.size == 0):
          mat_rec = mat
          label_rec.append(lists)
        else:
          mat_rec = np.hstack((mat_rec, mat))
          label_rec.append(lists)
      else:
        if (mat_train.size == 0):
          mat_train = mat
          label_train.append(lists)
        else:
          mat_train = np.hstack((mat_train, mat))
          label_train.append(lists)

  print('mat_train', mat_train , mat_train.shape)
  print('label_train', label_train)
  print('mat_rec', mat_rec , mat_rec.shape)
  print('label_rec', label_rec)

  # 存储处理完的训练数据集和标签
  # np.save('mat_train', mat_train)
  # np.save('label_train', label_train)
  # np.save('mat_rec', mat_rec)
  # np.save('label_rec', label_rec)
  return mat_train


def feature_cal(C):
  # 求出特征值eigenvalue，特征向量featurevector
  eigenvalue, featurevector = np.linalg.eig(C)
  # 定义一个临时矩阵用来存储前K个特征向量
  mat_temp = np.mat([])
  # 取前K个特征向量作为基
  mat_temp = featurevector[:,0]
  for i in range(1, K):
    mat_temp = np.hstack((mat_temp, featurevector[:,i]))
  mat_temp = np.array(mat_temp)
  np.save('feature_vector', mat_temp)

# 求特征脸并且输出
def feature_face_out(feature_face):
  for img_num in range(0, feature_face.shape[0]):
    savename = str(img_num + 1) + '.png'
    image_array = feature_face[img_num].reshape((112, 92))
    image_array = image_array / image_array.max()
    img = Image.fromarray(np.uint8(image_array * 255), 'L')
    img.save(os.path.join(savepath,savename))
    img.show()


#               shape[0]   shape[1]
# X                10304 * 280
# X_T                280 * 10304
# C                10304 * 10304
# feature_space    10304 * 50
# feature_face_all    50 * 280
# feature_face       280 * 10304

def PCA(X):
  # 每个维度去中心化 如果是拉成列那么行去中心化 反之则反之
  # mean(0)表示列平均值 mean(1)表示行平均值
  #np.save('X', X)
  meanMatrix = X.mean(1)
  #np.save('meanMatrix', meanMatrix)
  # 去中心化的数据集其实就是平均脸
  X = X - meanMatrix
  X_T = np.transpose(X)
  # C是原始协方差矩阵
  C = np.mat(((1 / X.shape[0]) * X * X_T))
  # 求协方差矩阵的单位特征向量和特征值
  #feature_cal(C)

  # 读取属于特征空间的基向量
  feature_space = np.mat(Read_mat())
  # 生成特征脸特征脸就是平均脸在K组正交基对应的特征空间上的投影
  feature_face_all = np.transpose(feature_space) * X
  # np.save('feature_face_all', np.array(feature_face_all))
  # 还原脸
  feature_face = np.transpose((np.transpose(feature_space)).I * feature_face_all)
  # 生成特征脸
  print('feature_face', feature_face)
  feature_face_out(feature_face)
  print('OK!')

# 读取特征向量的基
def Read_mat():
  feature_vector = np.load('feature_vector.npy')
  return feature_vector

if __name__ == "__main__":
  database = get_train_img()
  print(database.shape)
  PCA(database)

