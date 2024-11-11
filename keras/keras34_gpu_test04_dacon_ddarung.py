# keras30_MCP_save_04_dacon_ddarung.py 복사

# https://dacon.io/competitions/open/235576/overview/description (대회 사이트 주소)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import sklearn as sk
import time

#1. 데이터
path = "C://ai5//_data//dacon//따릉이//"

train_csv = pd.read_csv(path + "train.csv", index_col=0)    # 인덱스 없으면 index_col쓰면 안됨. 0은 0번째 줄 없앴다는 뜻이다.
# print(train_csv)    # [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
# print(submission_csv)   # [715 rows x 1 columns]

# print(train_csv.shape)  # (1459, 10)
# print(test_csv.shape) # (715, 9)
# print(submission_csv.shape) # (715, 1)

# print(train_csv.columns)    # 열의 이름을 알려달라는 수식. 
# # Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
# #        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
# #        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
# #       dtype='object')

# print(train_csv.info()) # Non-Null이 몇갠지 구멍난 데이터가 있는지 확인하는 수식.

# ################# 결측치 처리 1. 삭제 ################### 행은 다 같아야 함.
# # print(train_csv.isnull().sum()) 밑에 코드도 같은 코드다. isnull이나 isna나 똑같다.
# print(train_csv.isna().sum())   # 구멍난 데이터의 수를 알려달라.   

train_csv = train_csv.dropna()  # 구멍난 데이터를 삭제해달라는 수식
# print(train_csv.isna().sum())   # 잘 지워졌는지 확인.

# print(train_csv)    # [1328 rows x 10 columns]
# print(train_csv.isna().sum())   # 다시 확인
# print(train_csv.info()) # 다시 확인

# print(test_csv.info())  # 이제 test 파일도 정보확인

test_csv = test_csv.fillna(test_csv.mean()) # 구멍난 데이터를 평균값으로 채워달라는 뜻.
# print(test_csv.info()) # 715 non-nul /  확인.

x = train_csv.drop(['count'], axis=1)   # train_csv에서 count 지우는 수식을 만들고 있다. count 컬럼의 axis는 가로 1줄을 지운다. 행을 지운다. []안해도 나온다.
# print(x)    # [1328 rows x 9 columns] / 확인해봄.
y = train_csv['count']  # y는 count 열만 가지고 옴. y를 만들고 있다.
# print(y.shape)  # (1328,)   # 확인해봄.

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, 
                                                    random_state=4343,
                                                    shuffle=True,
                                                    ) # random_state=3454, 맛집 레시피 : 4343 / stratify=y

scaler = MinMaxScaler() # MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv=scaler.transform(test_csv)

# #2. 모델 구성
# model = Sequential()
# model.add(Dense(64, activation='relu', input_dim=9))   # activation='relu'
# model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='linear')) # activation='linear'

#2-2. 모델 구성 (함수형)
input1 = Input(shape=(9,))
dense1 = Dense(64, name='ys1', activation='relu')(input1)  # 레이어 이름도 변경가능, 성능에는 영향을 안 미친다.
dense2 = Dense(64, name='ys2', activation='relu')(dense1)
drop1 = Dropout(0.3)(dense2)
dense3 = Dense(32, name='ys3', activation='relu')(drop1)
drop2 = Dropout(0.3)(dense3)
dense4 = Dense(32, name='ys4', activation='relu')(drop2)
output1 = Dense(1)(dense4)
model = Model(inputs = input1, outputs = output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # metrics=['accuracy', 'acc', 'mse']
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
filepath = "".join([path, 'k32_dropout_dacon_ddarung', date, '_', filename])
# 생성 예: "./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"

########################### mcp 세이프 파일명 만들기 끗 ################

mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_olny=True, 
    filepath = filepath,
)

model.fit(x_train, y_train, epochs=1000, batch_size=8,
          verbose=1, validation_split=0.2,
          callbacks= [es, mcp]
          )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

y_submit = model.predict(test_csv)  # 예측한 값을 y_submit에 넣는다는 뜻.
# print(y_submit) # 확인 해봄.
# print(y_submit.shape)   # (715, 1) / 나왔음.

###################### submissinon.csv만들기 // count컬럼에 값만 넣으주면 돼. ##########

submission_csv['count'] = y_submit  # submission count 열에 y_submit을 넣겠다는 수식.
# print(submission_csv)   # 확인
# print(submission_csv.shape) #.shape확인.

submission_csv.to_csv(path + "submission_0729_1301.csv")    # 폴더 안에 파일로 만들겠다. 가로 안은 (저장 경로 + 파일명)이다.

print("r2스코어 : ", round(r2, 4))
print ("로스는 : ", loss)   # 확인하려고 마지막에 집어넣음. 
print("걸린시간 : ", round(end_time - start_time, 2), "초")

if(gpus):
    print("쥐피유 돈다!!!")
else:
    print("쥐피유 없다! xxxx")
    
# CPU: 걸린시간 :  5.74 초
# GPU: 걸린시간 :  34.11 초