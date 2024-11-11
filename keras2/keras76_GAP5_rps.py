from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255)

path_train = './_data/image/rps/'

start_time1 = time.time()

xy_train = train_datagen.flow_from_directory(
    path_train, 
    target_size=(80, 80), 
    batch_size=2520, 
    class_mode='categorical', # 다중분류 - 원핫도 되서 나와욤.
    # class_mode='binary',    # 이중분류
    # color_mode='sparse',     # 다중분류
    # class_mode='None',       # y값이 없다!!!
    
    # color_mode='grayscale',
    color_mode='rgb',
    shuffle=True,
)   # Found 2520 images belonging to 3 classes.
# print(xy_train[0][1]) -> 값
# [1. 1. 2. 2. 1. 2. 0. 1. 2. 0. 0. 0. 2. 2. 2. 0. 1. 0. 0. 0. 1. 2. 1. 0.  0. 1. 0. 0. 1. 1.]

# print(xy_train[0][0].shape) # (30, 100, 100, 1)
# print(xy_train[0][0].shape) # (30, 100, 100, 3)

x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.8, 
                                                    shuffle= True,
                                                    random_state=666)

end_time1 = time.time()

# 2. 모델
vgg16 = VGG16(include_top=False, input_shape=(80, 80, 3))
vgg16.trainable = True  # False # 동결건조

model = Sequential()
model.add(vgg16)
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start_time = time.time()

model.fit(x_train, y_train, epochs=10, batch_size=1, validation_split=0.3, verbose=1)

end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

print("76_rps_로스는 : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("걸린시간: ", round(end_time - start_time, 2), "초")

# model.add(Flatten())


# model.add(GlobalAveragePooling2D())