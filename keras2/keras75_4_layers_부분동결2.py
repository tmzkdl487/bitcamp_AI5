# keras75_3_layers_부분동결.py 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

model = VGG16(weights='imagenet', include_top=True,)
    
model.layers[20].trainable = False # <- 이런식으로 모델 레이어를 골라서 동결시킬 수 있음.    
     
model.summary()
# model.layers[17] 동결 전
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0

# model.layers[17] 동결 후
# 17  <keras.layers.convolutional.Conv2D object at 0x000001262C83D190>    block5_conv3  False
# Total params: 138,357,544
# Trainable params: 135,997,736
# Non-trainable params: 2,359,808

# model.layers[20] 동결 후
# Total params: 138,357,544
# Trainable params: 35,593,000
# Non-trainable params: 102,764,544

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['layer Type', 'Layer Name', 'Layer Trainale'])
print(results)

