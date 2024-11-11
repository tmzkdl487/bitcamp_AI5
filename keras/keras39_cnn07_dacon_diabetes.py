# keras30_MCP_save_07_dacon_diabetes.py 복사
# https://dacon.io/competitions/official/236068/overview/description

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten,MaxPooling2D 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터
path = 'C://ai5/_data/dacon/diabetes/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv)    # [652 rows x 9 columns]

test_csv = pd.read_csv(path + "test.csv", index_col = 0)
# print(test_csv) # [116 rows x 8 columns]

sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

# print(train_csv.shape, test_csv.shape, sample_submission_csv.shape) 
# # (652, 9) (116, 8) (116, 1)

# print(train_csv.info())

x = train_csv.drop(['Outcome'], axis=1)

y = train_csv['Outcome']

print(x.shape)  #(652, 8)


x = x.to_numpy()
x = x.reshape(652, 2, 2, 2)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=455)

# print(x_train.shape, y_train.shape) # (65, 8) (65,) 
# print(x_test.shape, y_test.shape)   # (587, 8) (587,)

# scaler = MaxAbsScaler() # MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# #2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(2, 2, 2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(16, input_shape=(32,)))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor = 'val_loss',
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
start_time = time.time()

model.fit(x_train, y_train, epochs=100, batch_size=1,
          verbose=1, validation_split=0.3, callbacks= [es, mcp])
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

y_pred = model.predict(x_test)
# print(y_pred[:20])
y_pred = np.round(y_pred)
# print(y_pred[:20])

accuracy_score = accuracy_score(y_test, y_pred)

# y_submit = model.predict(test_csv)

# y_submit = np.round(y_submit)

# sample_submission_csv['Outcome'] = y_submit

# sample_submission_csv.to_csv(path + "sample_submission_0729_1246_RobustScaler.csv")

print("ACC : ", round(loss[1],3))
print("로스 : ", loss[0])
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# if(gpus):
#     print("쥐피유 돈다!!!")
# else:
#     print("쥐피유 없다! xxxx")

# CPU: 걸린시간 :  6.4 초
# GPU: 걸린시간 :  58.28 초

# dnn 데이터 -> cnn데이터로 바꾸기
# ACC :  0.621 / 로스 :  5.842783451080322 / 걸린시간 :  22.22 초
