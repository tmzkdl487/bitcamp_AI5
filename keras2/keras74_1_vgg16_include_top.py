import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import os

# 경고 메시지 최소화
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
tf.config.set_visible_devices([], 'GPU')  # GPU 비활성화

tf.random.set_seed(333)
np.random.seed(333)

from tensorflow.keras.applications import VGG16

# model = VGG16()
# model.summary()

########################### 디폴트 VGG 모델 ########################
# model = VGG16()

### 디폴트 : 
# VGG16(weights='imagenet',
#               include_top=True,
#               input_shape=(224, 224, 3))

# model.summary()
# input_1 (InputLayer)        [(None, 224, 224, 3)]     0
# predictions (Dense)         (None, 1000)              4097000
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _________________________________________________________________

model = VGG16(# weights='imagenet',
              include_top=False,
              input_shape=(100, 100, 3))
model.summary()

############### include_top = False #####################
#1. FC layer 날려
#2. input_shape를 내가 하고싶은 데이터 shape로 맞춰!!!