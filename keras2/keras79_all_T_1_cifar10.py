# 01. VGG19
# 02. Xception
# 03. ResNet50
# 04. ResNet101
# 05. InceptionV3
# 06. InceptionResNetV2
# 07. DenseNet121
# 08. MobileNetV2
# 09. NasNetMobile
# 10. EfficeintNetB0

# GAP 써라!!!!!
# 기존거와 최고 성능 비교!!!!

# keras79_all_T_2_cifar100
# keras79_all_T_3_horse
# keras79_all_T_4_rps
# keras79_all_T_5_kaggle_cat_dog
# keras79_all_T_6_men_women

from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications import EfficientNetB0

from keras.datasets import cifar10
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
import time

from tensorflow.keras.layers import Resizing

import numpy as np
import tensorflow as tf

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# print("TensorFlow version:", tf.__version__)
# print("GPU Available: ", tf.test.is_built_with_cuda())
# print("GPU Devices: ", tf.config.list_physical_devices('GPU'))

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = tf.image.resize(x_train, (75, 75)).numpy()
x_test = tf.image.resize(x_test, (75, 75)).numpy()

# 데이터 전처리
x_train = x_train/255.
x_test = x_test/255.

# 2. 모델 리스트
models = [ 
        VGG19, 
        Xception, ResNet50, 
        #  ResNet101, 
        InceptionV3, 
        InceptionResNetV2,
        DenseNet121, 
        MobileNetV2, 
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
        model.add(Dense(10, activation='softmax'))
       

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
        
        print('=============================================')
        print('model 이름 : ', models.name)
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"훈련시간: {end_time - start_time:.2f}초")

# ===========VGG19 모델 테스트===========
# model 이름 :  vgg19
# Loss: 1.3375
# Accuracy: 0.5442
# 훈련시간: 25.58초
# 전체 가중치 수: 238
# 훈련 가능 가중치 수: 4
# ============================================
# model 이름 :  xception
# Loss: 0.8551
# Accuracy: 0.7103
# 훈련시간: 28.77초
# 전체 가중치 수: 322
# 훈련 가능 가중치 수: 4
# ============================================
# model 이름 :  resnet50
# Loss: 2.0436
# Accuracy: 0.2652
# 훈련시간: 32.39초
# 전체 가중치 수: 628   
# 훈련 가능 가중치 수: 4
# ============================================
# ResNet101 터짐
# ============================================
# model 이름 :  inception_v3
# Loss: 1.1795
# Accuracy: 0.5939
# 훈련시간: 44.71초
# 전체 가중치 수: 900
# 훈련 가능 가중치 수: 4
# ============================================
# model 이름 :  inception_resnet_v2
# Loss: 0.9794
# Accuracy: 0.6682
# 훈련시간: 95.25초
# 전체 가중치 수: 608
# 훈련 가능 가중치 수: 4
# ============================================
# model 이름 :  densenet121
# Loss: 0.8558
# Accuracy: 0.7003
# 훈련시간: 47.20초
# ============================================
# model 이름 :  mobilenetv2_1.00_224
# Loss: 1.0656
# Accuracy: 0.6289
# 훈련시간: 21.66초
# ============================================
# model 이름 :  inception_resnet_v2
# Loss: 0.9784
# Accuracy: 0.6694
# 훈련시간: 101.08초
# 전체 가중치 수: 608
# 훈련 가능 가중치 수: 4
# ============================================
# NasNetMobile 터짐
# ============================================
#  capability: 8.6
# 전체 가중치 수: 316
# 훈련 가능 가중치 수: 4
# ============================================
# model 이름 :  efficientnetb0
# Loss: 2.3027
# Accuracy: 0.1000
# 훈련시간: 41.73초