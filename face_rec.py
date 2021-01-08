'''
本文件用来识别
从文件夹中导入识别数据集进行识别
'''
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img_num = 120 # 选取多少图片进行识别 最大值是mat_rec矩阵的列数

def Read_mat():
  feature_vector = np.load('feature_vector.npy')
  return feature_vector

def Read_feature_all():
  feature_face_all = np.load('feature_face_all.npy')
  return feature_face_all

def Sparse():
  Sparse_vector = []
  mat = np.mat([])
  X = np.load('X.npy')
  label_train = np.load('label_train.npy')
  label_rec = np.load('label_rec.npy')
  mat_train = np.load('mat_train.npy')
  mat_rec = np.load('mat_rec.npy')
  mat = (np.mat(mat_train)).I * mat_rec
  temp = []

  sum = 0
  for i in range(0,img_num):
    sort_arr = [ ]
    temp = np.transpose(mat[:,i])
    List = [ ]
    for j in range(0,40):
      sum_weight = 0
      for z in range(j*7,(j+1)*7):
        sum_weight = sum_weight + mat[z,i]
      sum_weight = sum_weight / 7
      for k in range(0,7):
        List.append(sum_weight)

    List = np.array(List)
    sort_arr = np.argsort(-List)
    a = sort_arr[0]
    if (label_train[a] == label_rec[i]):
      sum = sum + 1

  rec_percent = float(sum / img_num)
  print('Sparse recognition:')
  print('The correct is: ', sum)
  print('correct_percent: {:.1f}%'.format(rec_percent * 100))

def PCA():
  # 读取去中心化值 训练集标签
  meanMatrix = np.load('meanMatrix.npy')
  label_train = np.load('label_train.npy')

  # 读取识别矩阵
  mat_rec = np.load('mat_rec.npy')
  label_rec = np.load('label_rec.npy')
  # 去中心化
  mat_rec = mat_rec - meanMatrix

  # 读取特征空间基
  feature_space = np.mat(Read_mat())
  # 读取特征脸训练数据
  feature_face_all = np.array(Read_feature_all())
  # 生成识别集矩阵
  feature_face = np.array(np.transpose(feature_space) * mat_rec)
  # 识别数据集一列为一个图片的投影
  sum = 0
  for j in range(0,img_num):
    dist = []
    for i in range(0,280):
      dist.append(np.linalg.norm(feature_face[:,j] - feature_face_all[:,i]))
    sort_arr = np.argsort(dist)
    #print(sort_arr)
    #print('The min distance is :', dist[sort_arr[0]], ' num is :', sort_arr[0])
    #print('The label is :', label_train[sort_arr[0]])
    if (label_train[sort_arr[0]] == label_rec[j]):
      sum = sum + 1
  rec_percent = float(sum / img_num)
  print('PCA recognition:')
  print('The correct is: ' , sum)
  print('correct_percent: {:.1f}%'.format(rec_percent*100))

def Sparse_PCA():
  temp = []
  # 读取去中心化值 训练集标签
  meanMatrix = np.load('meanMatrix.npy')
  label_train = np.load('label_train.npy')

  # 读取识别矩阵
  mat_rec = np.load('mat_rec.npy')
  label_rec = np.load('label_rec.npy')
  # 去中心化
  mat_rec = mat_rec - meanMatrix

  # 读取特征空间基
  feature_space = np.mat(Read_mat())
  # 读取特征脸训练数据
  feature_face_all = np.array(Read_feature_all())
  # 生成识别集矩阵
  feature_face = np.array(np.transpose(feature_space) * mat_rec)
  mat = (np.mat(feature_face_all)).I * feature_face
  sum = 0
  for i in range(0, img_num):
    sort_arr = []
    temp = np.transpose(mat[:, i])
    List = []
    for j in range(0, 40):
      sum_weight = 0
      for z in range(j * 7, (j + 1) * 7):
        sum_weight = sum_weight + mat[z, i]
      sum_weight = sum_weight / 7
      for k in range(0, 7):
        List.append(sum_weight)

    List = np.array(List)
    sort_arr = np.argsort(-List)
    a = sort_arr[0]
    if (label_train[a] == label_rec[i]):
      sum = sum + 1

  rec_percent = float(sum / img_num)
  print('Sparse_PCA recognition:')
  print('The correct is: ', sum)
  print('correct_percent: {:.1f}%'.format(rec_percent * 100))

if __name__ == "__main__":
  print('rec_num is: ', img_num)
  Sparse()
  PCA()
  Sparse_PCA()




