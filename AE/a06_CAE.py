# a3_ae2를 카피해서 모델 구성

# 인코더      28
#    conv     28
#    maxpool  14
#    conv     14 
#    maxpool  7

# 디코더
#    conv    7
#    UpSampling2D(2, 2) 14
#    conv               14
#    UpSampling2D(2, 2) 28
#    Conv(1, (3, 3))    -> (28, 28, 1)로 맹그러

# 시작!!! 맹그러!!!!

# a03_ae2_그림.py 카피

import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf

np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = np.expand_dims(x_train, axis=-1) 
x_test = np.expand_dims(x_test, axis=-1)   

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

# np.clip을 사용하여 값의 범위를 0과 1로 제한
x_train_noised = np.clip(x_train_noised,a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

print(x_train_noised.shape, x_test_noised.shape)    # (60000, 28, 28, 1) (10000, 28, 28, 1)

#2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

input_img = Input(shape=(28, 28, 1))

## 인코더
encoded1 = Conv2D(28, (2, 2), activation='relu', padding='same')(input_img)
encoded2 = MaxPooling2D((2, 2), padding='same')(encoded1)
encoded3 = Conv2D(14, (2, 2), activation='relu', padding='same')(encoded2)
encoded4 = MaxPooling2D((2, 2), padding='same')(encoded3)

## 디코더
decoded1 = Conv2D(7, (2, 2), activation='relu', padding='same')(encoded4)
decoded2 = UpSampling2D((2, 2))(decoded1)  
decoded3 = Conv2D(14, (2, 2), activation='relu', padding='same')(decoded2)
decoded4 = UpSampling2D((2, 2))(decoded3)  
decoded5 = Conv2D(1, (3, 3), padding='same')(decoded4)

autoencoder = Model(input_img, decoded5)

#3. 컴파일, 훈련
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train_noised, x_train, epochs=50, batch_size=128,
                validation_split=0.2)

#4. 평가, 예측
decoded_imgs = autoencoder.predict(x_test_noised)

############# 아무 생각없이 타자연습 ######################
import matplotlib.pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
      (ax11, ax12, ax13, ax14, ax15)) = \
       plt.subplots(3, 5, figsize=(20, 7))

# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(decoded_imgs.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

# 노이즈를 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0:
        ax.set_ylabel("NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(decoded_imgs[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()   