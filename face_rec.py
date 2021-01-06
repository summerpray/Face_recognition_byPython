import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def img_init():
  img = Image.open('D:/Facial_Recognition/Face_recognition/rec/7.pgm')
  img = np.array(img.convert('L'), 'f')
  arr = np.transpose(np.mat(img.flatten()))
  return arr

def Read_mat():
  a = np.load('feature_vector_noshape.npy')
  return a

def Read_feature_all():
  a = np.load('feature_face_all.npy')
  return a

if __name__ == "__main__":
  face = img_init()
  meanMatrix = np.load('meanMatrix.npy')
  face = face - meanMatrix
  #读取特征空间基
  feature_space = np.mat(Read_mat())
  #读取特征脸数据
  feature_face_all = np.array(Read_feature_all())
  print(feature_face_all.shape)
  feature_face = np.transpose(np.array(feature_space * face))
  dist = []
  for i in range(0,50):
    dist.append(np.linalg.norm(feature_face - feature_face_all[:,i]))
  print(dist)
  sort_arr = np.argsort(dist)
  print(sort_arr)
  print('The min distance is :',dist[sort_arr[0]], ' num is :' ,sort_arr[0])