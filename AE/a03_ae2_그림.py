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

hidden_size = 713      # pca 1.0
# hidden_size = 486      # pca 0.999
# hidden_size = 331      # pca 0.99
# hidden_size = 154        # pca 0.95

model = autoencoder(hidden_layer_size=hidden_size)

#3. 컴파일, 훈련
model.compile(optimizer='adam', loss='mse')
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(x_train_noised, x_train, epochs=30, batch_size=128,
                validation_split=0.2)

#4. 평가, 예측
decoded_imgs = model.predict(x_test_noised)

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