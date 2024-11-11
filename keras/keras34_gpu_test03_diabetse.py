# keras30_MCP_save_03_diabetse.py 복사

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import sklearn as sk
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target 

print(x)    
print(y)
print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=8989)

scaler = MaxAbsScaler() # MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# #2. 모델 구성
# model = Sequential()
# model.add(Dense(256, input_dim=10))
# model.add(Dropout(0.3))
# model.add(Dense(128))
# model.add(Dropout(0.3))
# model.add(Dense(64))
# model.add(Dropout(0.3))
# model.add(Dense(32))
# model.add(Dropout(0.3))
# model.add(Dense(5))
# model.add(Dropout(0.3))
# model.add(Dense(1))

#2-2. 모델 구성 (함수형)
input1 = Input(shape=(10,))
dense1 = Dense(256, name='ys1', activation='relu')(input1)  # 레이어 이름도 변경가능, 성능에는 영향을 안 미친다.
dense2 = Dense(128, name='ys2', activation='relu')(dense1)
drop1 = Dropout(0.3)(dense2)
dense3 = Dense(64, name='ys3', activation='relu')(drop1)
drop2 = Dropout(0.3)(dense3)
dense4 = Dense(32, name='ys4', activation='relu')(drop2)
output1 = Dense(1)(dense4)
model = Model(inputs = input1, outputs = output1) 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()

es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights=True,
)

########################### mcp 세이프 파일명 만들기 시작 ################
import datetime 
date = datetime.datetime.now()
print(date) # 2024-07-26 16:51:36.578483
print(type(date))
date = date.strftime("%m%d_%H%M")
print(date) # 0726 / 0726_1654
print(type(date))

path = './_save/keras32/'
filename = '{epoch:04d}-{val_loss:4f}.hdf5' # '1000-0.7777.hdf5'
filepath = "".join([path, 'k32_', date, '_', filename])
# 생성 예: "./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"

########################### mcp 세이프 파일명 만들기 끗 ################

mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_olny=True, 
    filepath = filepath,
)

model.fit(x_train, y_train, epochs=1000, batch_size=32, 
          verbose=0, validation_split=0.2,
          callbacks= [es, mcp]
          )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print ("로스 : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)
print("걸린시간 : ", round(end_time - start_time, 2), "초")

if(gpus):
    print("쥐피유 돈다!!!")
else:
    print("쥐피유 없다! xxxx")

# CPU: 걸린시간 :  1.02 초
# GPU: 걸린시간 :  1.78 초
