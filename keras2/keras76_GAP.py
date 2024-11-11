# keras74_2_vgg16_cifar10.py 카피

from keras.datasets import cifar10
from tensorflow.keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
import numpy as np
import tensorflow as tf
import os
import time

tf.random.set_seed(333)
np.random.seed(333)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train/255.
x_test = x_test/255.

#2. 모델
# VGG16 모델 불러오기
vgg16 = VGG16(#weights='imagenet', 
              include_top=False,
              input_shape=(224, 224, 3))

# vgg16.trainable = True # 가중치 동결

# 새 모델 정의
model = Sequential()
model.add(vgg16)  # VGG16의 기본 기능 사용
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  vgg16 (Functional)          (None, 7, 7, 512)         14714688

#  global_average_pooling2d (G  (None, 512)              0
#  lobalAveragePooling2D)

#  dense (Dense)               (None, 100)               51300

#  dense_1 (Dense)             (None, 100)               10100

#  dense_2 (Dense)             (None, 10)                1010

# =================================================================
# Total params: 14,777,098
# Trainable params: 14,777,098
# Non-trainable params: 0
# _________________________________________________________________
