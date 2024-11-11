# keras14_train_test1 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터  
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=0  
          validation_data=(x_val, y_val))   # 이 파일에서 요놈만 추가됨.)    

#.4 평가, 예측
print("++++++++++++++++++++++++++++++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)   # 데이터는 2개로 나눌 것이다.
results = model.predict([11])
print("로스 : ", loss )
print('[11]의 예측값 : ', results)

