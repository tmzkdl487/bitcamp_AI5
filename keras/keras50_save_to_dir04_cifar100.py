# keras49_augment4_cifar100.py 복사

from tensorflow.keras.datasets import mnist, fashion_mnist, cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

# x_train = x_train/255.
# x_test = x_test/255.

# x_train = x_train.reshape(50000,32*32*3)
# x_test = x_test.reshape(10000,32*32*3)

train_datagen = ImageDataGenerator(
    rescale=1./255,  # 스켈링한 데이터로 줘라, 수치화. 수치화만 하고 싶으면 밑에는 다 안써도 됨.
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.2,  # 평행이동 <- 위에 수평, 수직, 평행이동 데이터를 추가하면 8배의 데이터가 늘어난다.
    # height_shift_range=0.1, # 평행이동 수직
    rotation_range= 15,      # 정해진 각도만큼 이미지 회전 
    # zoom_range=1.2,         # 축소 또는 확대
    # shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환.
    fill_mode='nearest',    # 몇 개 더 있지만, 대표적으로 0도 있음. 너의 빈자리 비슷한 거로 채워줄께.
)

augment_size = 50000  # 증가시키다.

randidx = np.random.randint(x_train.shape[0], size=augment_size)    

x_augmented = x_train[randidx].copy()   # .copy()하면 메모리값을 새로 할당하기 때문에 원래 메모리값에 영향을 미치지 않는다. 메모리 안전빵.
y_augmented = y_train[randidx].copy()   #  x, y 5만개 준비됨.

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],         
    x_augmented.shape[1],          
    x_augmented.shape[2], 3) 

# print(x_augmented.shape)    # (50000, 32, 32, 3)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
    save_to_dir='c:/ai5/_data/_save_img/04_cifar100/'
).next()[0]
