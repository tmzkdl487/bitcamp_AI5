import os
os.environ['PATH'] = r"C:\Anaconda3\Library\bin;" + os.environ['PATH']

import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
np.random.seed(333)
tf.random.set_seed(333)

# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype("float32")/255.
x_test = x_test.reshape(10000, 28*28).astype("float32")/255.

# 평균 0, 표준편차 0.1인 정규분포 형태의 랜덤값 추가
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

# np.clip을 사용하여 값 제한
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(28*28,)))
    model.add(Dense(784, activation='sigmoid'))
    return model

list = [1, 8, 32, 64, 154, 331, 486, 713]
outputs = []

for i in list:
    model = autoencoder(hidden_layer_size=i)
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train_noised, x_train, epochs=1, batch_size=32, verbose=1)
    
    decoded_imgs = model.predict(x_test_noised)
    outputs.append(decoded_imgs)

# 3. 그래프 그리기
import matplotlib.pyplot as plt
import random

fig, axes = plt.subplots(len(list)+1, 5, figsize=(15, 15))

# 원본 데이터 포함
random_images = random.sample(range(x_test.shape[0]), 5)

# 첫 번째 행: 원본 데이터
for col_num, ax in enumerate(axes[0]):
    ax.imshow(x_test[random_images[col_num]].reshape(28, 28), cmap='gray')
    ax.set_title("Original")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 나머지 행: 모델 출력
for row_num, decoded_imgs in enumerate(outputs, start=1):
    for col_num, ax in enumerate(axes[row_num]):
        ax.imshow(decoded_imgs[random_images[col_num]].reshape(28, 28), cmap='gray')
        ax.set_title(f"Node {list[row_num-1]}")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()