# keras40_hamsu4_cifar100.py

from tensorflow.keras.datasets import mnist, fashion_mnist, cifar100
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D, Conv1D, MaxPool1D
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

x_train = x_train/255.
x_test = x_test/255.

# x_train = x_train.reshape(50000,32*32*3)
# x_test = x_test.reshape(10000,32*32*3)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1,1)   
y_test = y_test.reshape(-1,1)  
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)

# exit()

x_train = x_train.reshape(50000, 32, 32*3) 
x_test = x_test.reshape(10000, 32, 32*3) 

# print(x_train.shape, x_test.shape)  # (50000, 32, 96) (10000, 32, 96)

# exit()

#2. 모델
# model = Sequential()
# model.add(Dense(64, activation='relu', input_shape=(32*32*3,)))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(units=32, activation='relu')) 
# model.add(Dense(128, input_shape=(32,)))
# model.add(Dense(100, activation='softmax'))

#2-2. 모델 구성 (함수형)
# input1 = Input(shape=(32, 32, 3))
# Conv2D1 = Conv2D(128, (3,3), name='ys1',  activation='relu', strides=2, padding='same')(input1)  # 레이어 이름도 변경가능, 성능에는 영향을 안 미친다.
# MaxPooling2D1 = MaxPooling2D()(Conv2D1)
# Conv2D2 = Conv2D(64, (3,3), name='ys2',  activation='relu', strides=2, padding='same')(MaxPooling2D1)
# MaxPooling2D1 = MaxPooling2D()(Conv2D2)
# Conv2D3 = Conv2D(64, (2,2), name='ys3',  activation='relu', strides=2, padding='same')(MaxPooling2D1)
# Flatten1 = Flatten()(Conv2D3)
# Dense1 = Dense(32, activation='relu')(Flatten1)
# Dense2 = Dense(16, activation='relu')(Dense1)
# output1 = Dense(100, activation='softmax')(Dense2)
# model = Model(inputs = input1, outputs = output1)

#2. Conv1D 모델
model = Sequential()
model.add(Conv1D(10, kernel_size=2, input_shape=(32, 96))) # timesteps, features
model.add(Conv1D(10, 2))
model.add(Flatten())
model.add(Dense(20)) # RNN은 Dense와 바로 연결이 가능하다.
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['acc'])
start_time = time.time()

es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience= 10, 
    restore_best_weights= True
)

############ 세이프 파일명 만들기 시작 ############
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras35/'
filename = '{epoch:04d}-{val_loss:4f}.hdf5'
filepath = "".join([path, 'k35_06', date, '_', filename])
###### mcp 세이프 파일명 만들기 끗 ###############

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only=True,
    filepath = filepath,
)

model.fit(x_train, y_train, epochs=100, batch_size=513,
          validation_split=0.3, verbose=1, callbacks=[es, mcp])

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

y_pred = model.predict(x_test)

# y_test = y_test.to_numpy()

y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

acc = accuracy_score(y_test, y_pred)

print("로스는 : ", round(loss[0], 3))
print("ACC : ", round(loss[1], 3))
print("걸린시간: ", round(end_time - start_time, 2), "초")

# ACC :  0.235
# ACC :  로스는 :  3.9026308059692383 / ACC :  0.172 / 걸린시간:  516.04 초 <- 배치 500
# 로스는 :  4.570847034454346 / ACC :  0.108 / 걸린시간:  716.3 초
# 로스는 :  3.3694465160369873 / ACC :  0.205 / 걸린시간:  156.54 초

# strides=2, padding='same' 넣어서 성능 개선해보기
# 로스는 :  2.990478515625 / ACC :  0.289 / 걸린시간:  83.16 초
# 로스는 :  2.817396402359009 / ACC :  0.296 / 걸린시간:  149.98 초
# 로스는 :  2.835911512374878 / ACC :  0.303 / 걸린시간:  565.27 초 / batch_size=16
# 로스는 :  2.89253830909729 / ACC :  0.299 / 걸린시간:  978.04 초 / batch_size=8

# MaxPooling 넣어서 성능 개선해보기
# 로스는 :  2.5927178859710693 / ACC :  0.339 / 걸린시간:  424.72 초

# 데이터 쫙 피고 다시 돌림.
# 로스는 :  4.270289421081543 / ACC :  0.032 / 걸린시간:  123.36 초

# 모델 함수로 돌림.
# 로스는 :  0.009900020435452461 / ACC :  0.99 / 걸린시간:  12.42 초
# 로스는 :  0.009 / ACC :  0.228 / 걸린시간:  45.13 초

# 모델 Conv1D 
# 로스는 :  0.01 / ACC :  0.99 / 걸린시간:  41.6 초