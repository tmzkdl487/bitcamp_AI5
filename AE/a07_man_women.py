# keras45_7 ~ 등을 참고해서
# 남자, 여자 사진에 노이즈를 주고, (내 사진도 노이즈 만들고)
# 오토인코더로 피부 미백 훈련 가중치를 만든다.

# 그 가중치로 내 사진을 프레딕트해서
# 피부 미백 시킨다.

# 출력 이미지는 (원본, 노이즈, predict) 순으로 출력

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

import tensorflow as tf
# GPU 메모리 동적 할당 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import PIL.Image as Image

#1. 데이터

np_path = 'c:/ai5/_data/_save_npy/'


x_train = np.load(np_path + 'keras45_gender_04_x_train.npy')[:5000]

x_test = np.array(Image.open('C:/ai5/Bon_project/Thesis/01_RCNN/kimjihye.jpg').resize((100, 100))).reshape(1, 100, 100, 3) / 255.

# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.

# x_train = np.expand_dims(x_train, axis=-1) 
# x_test = np.expand_dims(x_test, axis=-1)   

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

# np.clip을 사용하여 값의 범위를 0과 1로 제한
x_train_noised = np.clip(x_train_noised,a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)


#2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size, (3,3),input_shape=(100, 100,3), padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(hidden_layer_size, (3,3), padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(hidden_layer_size, (3,3), padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(hidden_layer_size, (3,3), padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(3, (3,3), activation='sigmoid', padding='same'))
    return model

# hidden_size = 713       # pca 1.0
# hidden_size = 486       # pca 0.999
# hidden_size = 331       # pca 0.99
hidden_size = 128       # pca 0.95

model = autoencoder(hidden_layer_size=hidden_size)  

#3. 컴파일, 훈련
model.compile(optimizer='adam', loss='mse')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    x_train_noised, x_train,  # 훈련 데이터 제한
    epochs=10,  # 에포크 10으로 설정
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)

#4. 평가, 예측
decoded_imgs = model.predict(x_test_noised)

import matplotlib.pyplot as plt
import random

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# 원본 이미지
ax1.imshow(x_test[0])
ax1.set_title("Original")
ax1.axis('off')

# 노이즈 추가 이미지
ax2.imshow(x_test_noised[0])
ax2.set_title("Noised")
ax2.axis('off')

# 예측 이미지
ax3.imshow(decoded_imgs [0])
ax3.set_title("Predicted")
ax3.axis('off')

plt.tight_layout()
plt.show()