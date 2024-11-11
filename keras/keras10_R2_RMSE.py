# keras09_scatter2. 카피
# [검색] R2 Score (결정계수)
'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.75, random_state=1234)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense (33))
model.add(Dense (33))
model.add(Dense (33))
model.add(Dense (1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스: ", loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

# 로스:  5.074116230010986 r2스코어 :  0.8641831921367648
# 로스:  13.100679397583008 r2스코어 :  0.6493393922483062
# 로스:  6.318336009979248 r2스코어 :  0.8015597675671038

'''
########################## R2 와 RMSE ######################

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.75, random_state=1234)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense (33))
model.add(Dense (33))
model.add(Dense (33))
model.add(Dense (1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스: ", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)

print("RMSE : ", rmse)

# 로스:  7.406783103942871 / r2스코어 :  0.7673749042948712 / RMSE :  2.721540565057097