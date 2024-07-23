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

# x_train2 = np.load(np_path + 'keras45_07_gender_x_train.npy')
# y_train2 = np.load(np_path + 'keras45_07_gender_y_train.npy')

# # 개별 파일 불러오기
x_train = np.load(np_path + 'keras45_gender_04_x_train.npy')
y_train = np.load(np_path + 'keras45_gender_04_y_train.npy')
x_test = np.load(np_path + 'keras45_gender_04_x_test.npy')
y_test = np.load(np_path + 'keras45_gender_04_y_test.npy')

# x_train = np.repeat(x_train, 3, axis=-1)
# x_test = np.repeat(x_test, 3, axis=-1)

# 데이터 스케일링
x_train = x_train / 255.0
x_test = x_test / 255.0

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
        # NASNetMobile, 
        EfficientNetB0
]

# 3. 모델 생성 및 훈련
for i in models:    
        models = i(
            weights='imagenet',
            include_top=False,
            input_shape=(100, 100, 3)) 
        models.trainable = False

        # 모델 구성
        model = Sequential()
        model.add(models)
        model.add(GlobalAveragePooling2D()),
        model.add(Dense(32, activation='relu')),
        model.add(Dense(1, activation='softmax'))
       

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
        loss, acc = model.evaluate(x_val, y_val, verbose=0)
       
        print('==================================================')
        print('model 이름 : ', models.name)
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"훈련시간: {end_time - start_time:.2f}초")
        
# 전체 가중치 수: 36
# 훈련 가능 가중치 수: 4
# ==================================================
# model 이름 :  vgg19
# Loss: 0.6510
# Accuracy: 0.3537
# 훈련시간: 17.54초
# 전체 가중치 수: 238
# 훈련 가능 가중치 수: 4
# ==================================================
# model 이름 :  xception
# Loss: 0.6489
# Accuracy: 0.3537      
# 훈련시간: 15.19초
# 전체 가중치 수: 322
# 훈련 가능 가중치 수: 4
# ==================================================
# model 이름 :  resnet50
# Loss: 0.6683
# Accuracy: 0.3537
# 훈련시간: 16.60초
# 전체 가중치 수: 628
# 훈련 가능 가중치 수: 4
# =================================================
# model 이름 :  resnet101
# Loss: 0.6504
# Accuracy: 0.3537
# 훈련시간: 27.18초
# 전체 가중치 수: 380
# 훈련 가능 가중치 수: 4
# ==================================================
# model 이름 :  inception_v3
# Loss: 0.6236
# Accuracy: 0.3537
# 훈련시간: 20.82초
# 전체 가중치 수: 900
# 훈련 가능 가중치 수: 4
# ==================================================
# model 이름 :  inception_resnet_v2
# Loss: 0.6489
# Accuracy: 0.3537
# 훈련시간: 46.91초
# 전체 가중치 수: 608
# 훈련 가능 가중치 수: 4
# ==================================================
# model 이름 :  densenet121
# Loss: 0.6473
# Accuracy: 0.3537
# 훈련시간: 26.44초
# ==================================================
# model 이름 :  mobilenetv2_1.00_224
# Loss: 0.6493
# Accuracy: 0.3537
# 훈련시간: 13.27초
# ==================================================
# NasNetMobile 터짐
# ==================================================
# model 이름 :  efficientnetb0
# Loss: 0.6511
# Accuracy: 0.3537
# 훈련시간: 21.78초