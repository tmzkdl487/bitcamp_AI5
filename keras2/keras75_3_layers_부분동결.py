# keras75_2_layers.py 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))

vgg16.trainable = False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

#1. 전체동결
# model.trainable = False # 전체 모델 동결

#2. 전체동결
# for layer in model.layers: 
#     layer.trainable = False

#3. 부분동결
# print(model.layers)
# [<keras.engine.functional.Functional object at 0x00000183002F74C0>, <= vgg16
# <keras.layers.core.flatten.Flatten object at 0x00000183002FF1F0>,   <= Flatten
# <keras.layers.core.dense.Dense object at 0x0000018300568B80>,       <= Dense
# <keras.layers.core.dense.Dense object at 0x000001830056E880>]       <= Dense

# print(model.layers[0])  # <keras.engine.functional.Functional object at 0x000001FB004D84C0> <= vgg16
# print(model.layers[1])  # <keras.layers.core.flatten.Flatten object at 0x00000294B154F1F0>  <= Flatten
# print(model.layers[2])  # <keras.layers.core.dense.Dense object at 0x000002B702549BB0>      <= Dense
    
model.layers[2].trainable = False # <- 이런식으로 모델 레이어를 골라서 동결시킬 수 있음.    
     
model.summary()

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['layer Type', 'Layer Name', 'Layer Trainale'])
print(results)

# 0  <keras.engine.functional.Functional object at 0x000001673AD784F0>  vgg16      False
# 1  <keras.layers.core.flatten.Flatten object at 0x000001673AD80220>   flatten    True
# 2  <keras.layers.core.dense.Dense object at 0x000001673ADC9BB0>       dense      False
# 3  <keras.layers.core.dense.Dense object at 0x000001673ADD08B0>       dense_1    True
