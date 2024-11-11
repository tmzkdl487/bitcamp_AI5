import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.trainable = False # 전체 모델 동결

model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  vgg16 (Functional)          (None, 1, 1, 512)         14714688

#  flatten (Flatten)           (None, 512)               0

#  dense (Dense)               (None, 100)               51300

#  dense_1 (Dense)             (None, 10)                1010

# =================================================================
# Total params: 14,766,998
# Trainable params: 0
# Non-trainable params: 14,766,998
# _________________________________________________________________

print(len(model.weights))           # 30
print(len(model.trainable_weights)) # 0

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['layer Type', 'Layer Name', 'Layer Trainale'])
print(results)
#                                                           layer Type Layer Name  Layer Trainale
# 0  <keras.engine.functional.Functional object at 0x000001770BEC85B0>  vgg16      False
# 1  <keras.layers.core.flatten.Flatten object at 0x000001770BEBA9D0>   flatten    False
# 2  <keras.layers.core.dense.Dense object at 0x000001770BF18B20>       dense      False
# 3  <keras.layers.core.dense.Dense object at 0x000001770BF1F820>       dense_1    False

