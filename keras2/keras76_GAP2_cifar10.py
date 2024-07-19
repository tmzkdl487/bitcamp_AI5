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

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train/255.
x_test = x_test/255.

#2. 모델
# VGG16 모델 불러오기

vgg16 = VGG16(#weights='imagenet', 
              include_top=False, input_shape=(32, 32, 3))

vgg16.trainable = False # 동결건조

# 새 모델 정의
model = Sequential()
model.add(vgg16)  # VGG16의 기본 기능 사용
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
start_time = time.time()

model.fit(x_train, y_train, epochs=10, batch_size=1,
          validation_split=0.3, verbose=1)

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

y_pred = model.predict(x_test)

print("60_cifar10_로스는 : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("걸린시간: ", round(end_time - start_time, 2), "초")

# model.add(Flatten)
# 60_cifar10_로스는 :  1.2988202571868896
# ACC :  0.544
# 걸린시간:  1458.1 초

# model.add(GlobalAveragePooling2D())
# 60_cifar10_로스는 :  1.2988202571868896
# ACC :  0.544
# 걸린시간:  1485.66 초

