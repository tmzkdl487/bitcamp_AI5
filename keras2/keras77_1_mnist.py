from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Conv1D, MaxPool1D, GlobalAveragePooling2D 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#2. Conv1D 모델
model = Sequential()
model.add(Conv1D(10, kernel_size=2, input_shape=(28, 28))) # timesteps, features
model.add(Conv1D(10, 2))
model.add(GlobalAveragePooling2D())
model.add(Dense(20)) # RNN은 Dense와 바로 연결이 가능하다.
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['acc'])   # acc넣어야 분류일 경우 잘 맞는지 확인할 수 있음.
start_time = time.time()

model.fit(x_train, y_train, epochs=10, batch_size= 500,
          validation_split=0.3, verbose=1)

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)  

y_pred = model.predict(x_test)

print("로스는 : ", round(loss[0], 3))
print("걸린시간 : ", round(end_time - start_time, 2), "초")
