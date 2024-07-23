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

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255)

path_train = './_data/image/rps/'

start_time1 = time.time()

xy_train = train_datagen.flow_from_directory(
    path_train, 
    target_size=(75, 75), 
    batch_size=2520, 
    class_mode='categorical', # 다중분류 - 원핫도 되서 나와욤.
    # class_mode='binary',    # 이중분류
    # color_mode='sparse',     # 다중분류
    # class_mode='None',       # y값이 없다!!!
    
    # color_mode='grayscale',
    color_mode='rgb',
    shuffle=True,
)   

x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.8, 
                                                    shuffle= True,
                                                    random_state=666)

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
            input_shape=(224, 224, 3)) 
        models.trainable = False

        # 모델 구성
        model = Sequential()
        model.add(models)
        model.add(GlobalAveragePooling2D()),
        model.add(Dense(32, activation='relu')),
        model.add(Dense(3, activation='softmax'))
       

        # 모델 컴파일
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
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
        
# ==================================================
# model 이름 :  xception
# Loss: 0.0324
# Accuracy: 1.0000
# 훈련시간: 3.06초
# 전체 가중치 수: 322
# 훈련 가능 가중치 수: 4
# ==================================================
# model 이름 :  resnet50
# Loss: 0.7199
# Accuracy: 0.7817
# 훈련시간: 3.56초
# 전체 가중치 수: 628
# 훈련 가능 가중치 수: 4
# ==================================================
# model 이름 :  resnet101
# Loss: 0.6714
# Accuracy: 0.8294
# 훈련시간: 5.09초
# 전체 가중치 수: 380
# 훈련 가능 가중치 수: 4
# ==================================================
# model 이름 :  inception_v3
# Loss: 0.0101
# Accuracy: 1.0000
# 훈련시간: 5.06초
# 전체 가중치 수: 900
# 훈련 가능 가중치 수: 4
# ==================================================
# model 이름 :  inception_resnet_v2
# Loss: 0.0313
# Accuracy: 0.9960
# 훈련시간: 9.67초
# 전체 가중치 수: 608
# 훈련 가능 가중치 수: 4
# ==================================================
# model 이름 :  densenet121
# Loss: 0.0133
# Accuracy: 0.9960
# 훈련시간: 6.26초
# WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.
# 전체 가중치 수: 264
# 훈련 가능 가중치 수: 4
# ==================================================
# model 이름 :  mobilenetv2_1.00_224
# Loss: 0.0347
# Accuracy: 0.9921
# 훈련시간: 2.78초
# ==================================================
# model 이름 : NasNetMobile 터짐
# ==================================================
# model 이름 :  efficientnetb0
# Loss: 1.1092
# Accuracy: 0.3373
# 훈련시간: 6.86초
