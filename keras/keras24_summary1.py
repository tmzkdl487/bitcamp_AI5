from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

model.summary()
# input_dim = 1
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 3)                 6  # 3 (노드) + 1(바이어스) = 4 x 

#  dense_1 (Dense)             (None, 4)                 16 

#  dense_2 (Dense)             (None, 3)                 15

#  dense_3 (Dense)             (None, 1)                 4

# =================================================================
# Total params: 41
# Trainable params: 41
# Non-trainable params: 0

# input_dim = 2
# Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 3)                 9

#  dense_1 (Dense)             (None, 4)                 16

#  dense_2 (Dense)             (None, 3)                 15

#  dense_3 (Dense)             (None, 1)                 4

# =================================================================
# Total params: 44
# Trainable params: 44
# Non-trainable params: 0
# _________________________________________________________________
