from keras.datasets import cifar100
from tensorflow.keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, Conv2D, GlobalAveragePooling2D
import numpy as np
import tensorflow as tf
import os
import time

tf.random.set_seed(333)
np.random.seed(333)

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train/255.
x_test = x_test/255.

#2. Conv1D 모델
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # Conv2D로 변경
model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(20)) # RNN은 Dense와 바로 연결이 가능하다.
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])   # acc넣어야 분류일 경우 잘 맞는지 확인할 수 있음.
start_time = time.time()

model.fit(x_train, y_train, epochs=10, batch_size= 500,
          validation_split=0.3, verbose=1)

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)  

y_pred = model.predict(x_test)

print("77_cifar100_로스는 : ", round(loss[0], 3))
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# model.add(Flatten())
# 77_cifar100_로스는 :  3.387
# 걸린시간 :  11.58 초

# model.add(GlobalAveragePooling2D())
# 77_cifar100_로스는 :  4.028
# 걸린시간 :  11.32 초