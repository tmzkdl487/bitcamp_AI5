from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications import EfficientNetB0

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import time

import numpy as np
import tensorflow as tf

tf.random.set_seed(337)
np.random.seed(337)

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# 1. 데이터 불러오기
np_path = 'c:/ai5/_data/_save_npy/'  # npy 데이터 경로 설정

# 개별 파일 불러오기
x_train_1 = np.load(np_path + 'keras43_01_x_train.npy')
x_train_2 = np.load(np_path + 'keras49_image_cat_dog_01_x_train.npy')

y_train_1 = np.load(np_path + 'keras43_01_y_train.npy')
y_train_2 = np.load(np_path + 'keras49_image_cat_dog_01_y_train.npy')

x_test = np.load(np_path + 'keras43_01_x_test.npy')
y_test = np.load(np_path + 'keras43_01_y_test.npy')

# 데이터를 합치기
x_train = np.concatenate((x_train_1, x_train_2), axis=0)
y_train = np.concatenate((y_train_1, y_train_2), axis=0)

# 데이터 스케일링
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = tf.image.resize(x_train, (224, 224)).numpy()
x_test = tf.image.resize(x_test, (224, 224)).numpy()

# 데이터 분할
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.75, random_state=337)

# 2. 모델 리스트
models = [ 
        # VGG19, 
        # Xception, ResNet50, 
        # ResNet101, 
        # InceptionV3, 
        # InceptionResNetV2,
        # DenseNet121, 
        # MobileNetV2, 
        NASNetMobile, 
        # EfficientNetB0
]

# 3. 모델 생성 및 훈련
for i in models:    
        models = i(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3))
        models.trainable = False

        # 모델 구성
        model = Sequential()
        model.add(models)
        model.add(GlobalAveragePooling2D()),
        model.add(Dense(32, activation='relu')),
        model.add(Dense(1, activation='sigmoid'))
       

        # 모델 컴파일
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # 모델 정보 출력
        print("전체 가중치 수:", len(model.weights))
        print("훈련 가능 가중치 수:", len(model.trainable_weights))

        # EarlyStopping 설정
        es = EarlyStopping(
            monitor='val_loss',
            patience=8,
            mode='min',
            restore_best_weights=True
        )

        # 훈련 시작 시간
        start_time = time.time()

        # 모델 훈련
        history = model.fit(
            x_train, y_train,
            batch_size=16,
            epochs=1,
            validation_split=0.2,
            callbacks=[es],
            verbose=0
        )

        # 훈련 종료 시간
        end_time = time.time()

        # 모델 평가
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
       
        print('==================================================')
        print('model 이름 : ', models.name)
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"훈련시간: {end_time - start_time:.2f}초")
        
# 전체 가중치 수: 36
# 훈련 가능 가중치 수: 4

# ==================================================
# model 이름 :  vgg19
# Loss: 0.6309
# Accuracy: 1.0000
# 훈련시간: 19.44초
# 전체 가중치 수: 238   
# 훈련 가능 가중치 수: 4
# ==================================================
# model 이름 :  xception
# Loss: 0.6889
# Accuracy: 0.9998
# 훈련시간: 20.97초
# 전체 가중치 수: 322
# 훈련 가능 가중치 수: 4
# ==================================================
# model 이름 :  resnet50
# Loss: 0.6917
# Accuracy: 1.0000
# 훈련시간: 23.71초
# 전체 가중치 수: 628
# 훈련 가능 가중치 수: 4
# ==================================================
# model 이름 :  resnet101
# Loss: 0.6915
# Accuracy: 1.0000
# 훈련시간: 41.21초
# 전체 가중치 수: 380
# 훈련 가능 가중치 수: 4
# ==================================================
# model 이름 :  inception_v3
# Loss: 0.6781
# Accuracy: 0.9365
# 훈련시간: 31.84초
# 전체 가중치 수: 900
# 훈련 가능 가중치 수: 4
# ==================================================
# model 이름 :  inception_resnet_v2
# Loss: 0.6756
# Accuracy: 0.9986
# 훈련시간: 69.44초
# 전체 가중치 수: 608
# 훈련 가능 가중치 수: 4
# ==================================================
# model 이름 :  densenet121
# Loss: 0.6895
# Accuracy: 1.0000
# 훈련시간: 36.49초
# WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.
# 전체 가중치 수: 264
# 훈련 가능 가중치 수: 4
# ==================================================
# model 이름 :  mobilenetv2_1.00_224
# Loss: 0.6887
# Accuracy: 1.0000
# 훈련시간: 19.22초
# ==================================================
# NASNetMobile 
# ValueError: When setting `include_top=True` and loading `imagenet` weights, `input_shape` should be (224, 224, 3).  Received: input_shape=(80, 80, 3)
# 터짐.
# ==================================================
# model 이름 :  efficientnetb0
# Loss: 0.6913
# Accuracy: 1.0000
# 훈련시간: 32.82초