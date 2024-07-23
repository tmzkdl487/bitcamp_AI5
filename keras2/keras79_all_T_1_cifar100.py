from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications import EfficientNetB0

from keras.datasets import cifar100
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
import time

import numpy as np
import tensorflow as tf

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# print("TensorFlow version:", tf.__version__)
# print("GPU Available: ", tf.test.is_built_with_cuda())
# print("GPU Devices: ", tf.config.list_physical_devices('GPU'))

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = tf.image.resize(x_train, (75, 75)).numpy()
x_test = tf.image.resize(x_test, (75, 75)).numpy()

# 데이터 전처리
x_train = x_train/255.
x_test = x_test/255.

# 2. 모델 리스트
models = [ 
        #  VGG19, 
        # Xception, ResNet50, 
        # ResNet101, 
        # InceptionV3, 
        # InceptionResNetV2,
        # DenseNet121, 
        # MobileNetV2, 
        # NASNetMobile, 
        EfficientNetB0
]

# 3. 모델 생성 및 훈련
for i in models:    
        models = i(
            weights='imagenet',
            include_top=False,
            input_shape=(75, 75, 3)) 
        models.trainable = False

        # 모델 구성
        model = Sequential()
        model.add(models)
        model.add(GlobalAveragePooling2D()),
        model.add(Dense(32, activation='relu')),
        model.add(Dense(100, activation='softmax'))
       

        # 모델 컴파일
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # 모델 정보 출력
        print("전체 가중치 수:", len(model.weights))
        print("훈련 가능 가중치 수:", len(model.trainable_weights))

        # EarlyStopping 설정
        es = EarlyStopping(
            monitor='val_loss',
            patience=3,
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
        loss, acc = model.evaluate(x_test, y_test)
       
        print('model 이름 : ', models.name)
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"훈련시간: {end_time - start_time:.2f}초")
        
# 전체 가중치 수: 36
# 훈련 가능 가중치 수: 4
# ============================================
# model 이름 :  vgg19
# Loss: 3.3107
# Accuracy: 0.1995
# 훈련시간: 26.28초
# 전체 가중치 수: 238
# 훈련 가능 가중치 수: 4
# ============================================
# model 이름 :  xception
# Loss: 2.4174
# Accuracy: 0.3932
# 훈련시간: 28.04초
# 전체 가중치 수: 322
# 훈련 가능 가중치 수: 4
# ============================================
# model 이름 :  resnet50
# Loss: 4.5302
# Accuracy: 0.0196
# 훈련시간: 33.25초
# 전체 가중치 수: 628
# 훈련 가능 가중치 수: 4
# ============================================
# ResNet101 터짐
# ============================================
# model 이름 :  inception_v3
# Loss: 2.9220
# Accuracy: 0.2715
# 훈련시간: 44.16초
# 전체 가중치 수: 900
# 훈련 가능 가중치 수: 4
# ============================================
# model 이름 :  inception_resnet_v2
# Loss: 2.6935
# Accuracy: 0.3315
# 훈련시간: 94.18초
# 전체 가중치 수: 608
# 훈련 가능 가중치 수: 4
# ============================================
# model 이름 :  densenet121
# Loss: 2.4167
# Accuracy: 0.3676
# 훈련시간: 46.46초
# WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.
# 전체 가중치 수: 264
# 훈련 가능 가중치 수: 4
# ============================================
# model 이름 :  mobilenetv2_1.00_224
# Loss: 3.0022
# Accuracy: 0.2243
# 훈련시간: 24.41초
# ============================================
# NASNetMobile
# ValueError: When setting `include_top=True` and loading `imagenet` weights, `input_shape` should be (224, 224, 3).  Received: input_shape=(75, 75, 3)
# ============================================
# model 이름 :  efficientnetb0
# Loss: 4.6053
# Accuracy: 0.0100
# 훈련시간: 42.46초