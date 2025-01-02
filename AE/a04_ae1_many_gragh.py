# a03_ae1_pca.py 카피

import os
os.environ['PATH'] = r"C:\Anaconda3\Library\bin;" + os.environ['PATH']

import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf

np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.
                                            # 평균 0, 표편 0.1인 정규분포!!
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

print(x_train_noised.shape, x_test_noised.shape)        # (60000, 784) (10000, 784)
print(np.max(x_train), np.min(x_test))                  # 1.0 0.0
print(np.max(x_train_noised), np.min(x_test_noised))    # 1.506013411202829 -0.5281790150375157

# np.clip을 사용하여 값의 범위를 0과 1로 제한
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

print(np.max(x_train_noised), np.min(x_train_noised))  # 1.0 0.0
print(np.max(x_test_noised), np.min(x_test_noised))    # 1.0 0.0

#2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(28*28,)))
    model.add(Dense(784, activation='sigmoid'))
    return model

model_01 = autoencoder(hidden_layer_size=1)
model_08 = autoencoder(hidden_layer_size=8)
model_32 = autoencoder(hidden_layer_size=32)
model_64 = autoencoder(hidden_layer_size=64)
model_154 = autoencoder(hidden_layer_size=154) # PCA 95% 성능
model_331 = autoencoder(hidden_layer_size=331) # PCA 99% 성능
model_486 = autoencoder(hidden_layer_size=486) # PCA 99.9% 성능
model_713 = autoencoder(hidden_layer_size=713) # PCA 99.9% 성능
 
#3. 컴파일, 훈련
print("======================= node 1개 시작==================================")
model_01.compile(optimizer='adam', loss='mse')
model_01.fit(x_train_noised, x_train, epochs=1, batch_size=32, verbose=0)

print("======================= node 8개 시작==================================")
model_08.compile(optimizer='adam', loss='mse')
model_08.fit(x_train_noised, x_train, epochs=1, batch_size=32, verbose=0)

print("======================= node 32개 시작==================================")
model_32.compile(optimizer='adam', loss='mse')
model_32.fit(x_train_noised, x_train, epochs=1, batch_size=32, verbose=0)

print("======================= node 64개 시작==================================")
model_64.compile(optimizer='adam', loss='mse')
model_64.fit(x_train_noised, x_train, epochs=1, batch_size=32, verbose=0)

print("======================= node 154개 시작==================================")
model_154.compile(optimizer='adam', loss='mse')
model_154.fit(x_train_noised, x_train, epochs=1, batch_size=32, verbose=0)

print("======================= node 331개 시작==================================")
model_331.compile(optimizer='adam', loss='mse')
model_331.fit(x_train_noised, x_train, epochs=1, batch_size=32, verbose=0)

print("======================= node 486개 시작==================================")
model_486.compile(optimizer='adam', loss='mse')
model_486.fit(x_train_noised, x_train, epochs=1, batch_size=32, verbose=0)

print("======================= node 713개 시작==================================")
model_713.compile(optimizer='adam', loss='mse')
model_713.fit(x_train_noised, x_train, epochs=1, batch_size=32, verbose=0)

#4. 평가, 예측
decoded_imags_01 = model_01.predict(x_test_noised)
decoded_imags_08 = model_08.predict(x_test_noised)
decoded_imags_32 = model_32.predict(x_test_noised)
decoded_imags_64 = model_64.predict(x_test_noised)
decoded_imags_154 = model_154.predict(x_test_noised)
decoded_imags_331 = model_331.predict(x_test_noised)
decoded_imags_486 = model_486.predict(x_test_noised)
decoded_imags_713 = model_713.predict(x_test_noised)

################## 아무생각없이 타자연습 ###############################
import matplotlib.pyplot as plt
import random

fig, axes = plt.subplots(9, 5, figsize=(15, 15))

random_imags = random.sample(range(decoded_imags_01.shape[0]), 5)
outputs = [x_test, decoded_imags_01, decoded_imags_08, decoded_imags_32, decoded_imags_64, 
           decoded_imags_154, decoded_imags_331, decoded_imags_486, decoded_imags_713]

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_imags[col_num]].reshape(28, 28),
                  cmap='gray')
        # ax.set_yulabel(outputs[row_num])  # 이거 아니네 알아서 수정해!!!
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()
