# a02_ae_noised.py 카피

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
x_train_noised = np.clip(x_train_noised,a_min=0, a_max=1)
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

# PCA 1.0 hidden size: 682
# PCA 0.999 hidden size: 486
# PCA 0.99 hidden size: 331
# PCA 0.95 hidden size: 154

# PCA 1.0 hidden size: 713
# PCA 0.999 hidden size: 486
# PCA 0.99 hidden size: 331
# PCA 0.95 hidden size: 154

# hidden_size = 713      # pca 1.0
# hidden_size = 486      # pca 0.999
# hidden_size = 331      # pca 0.99
hidden_size = 154        # pca 0.95

model = autoencoder(hidden_layer_size=hidden_size)

#3. 컴파일, 훈련
model.compile(optimizer='adam', loss='mse')
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(x_train_noised, x_train, epochs=30, batch_size=128,
                validation_split=0.2)

#4. 평가, 예측
decoded_imgs = model.predict(x_test_noised)

import matplotlib.pyplot as plt
n=10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_noised[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()

# 챗 GPT
# PCA 1.0 hidden size: 682
# PCA 0.999 hidden size: 486
# PCA 0.99 hidden size: 331
# PCA 0.95 hidden size: 154

# 선생님
# PCA 1.0 hidden size: 713
# PCA 0.999 hidden size: 486
# PCA 0.99 hidden size: 331
# PCA 0.95 hidden size: 154