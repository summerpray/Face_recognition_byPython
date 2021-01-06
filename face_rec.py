import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def img_init():
  img = Image.open('D:/Facial_Recognition/Face_recognition/rec/10.pgm')
  img = np.array(img.convert('L'), 'f')
  arr = np.transpose(np.mat(img.flatten()))
  return arr

def Read_mat():
  a = np.load('feature_vector_7.npy')
  return a

def Read_feature_all():
  a = np.load('feature_face_all_7.npy')
  return a

if __name__ == "__main__":
  meanMatrix = np.load('meanMatrix_7.npy')

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
  for j in range(0,120):
    dist = []
    for i in range(0,280):
      dist.append(np.linalg.norm(feature_face[:,j] - feature_face_all[:,i]))
    sort_arr = np.argsort(dist)
    #print(sort_arr)
    #print('The min distance is :', dist[sort_arr[0]], ' num is :', sort_arr[0])
    #print('The label is :', label_train[sort_arr[0]])
    if (label_train[sort_arr[0]] == label_rec[j]):
      sum = sum + 1
  rec_percent = float(sum / 120)
  print('rec_num is: ', 120)
  print('The correct is: ' , sum)
  print('correct_percent: {:.1f}%'.format(rec_percent*100))
