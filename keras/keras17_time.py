# keras15_validation4_split 복사

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time # 통상 패키지라고 한다.

#1. 데이터  
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(
x, y, train_size=0.65, shuffle=True, random_state=133
)

print(x_train, y_train)
print(x_test, y_train)

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=1, 
          verbose=1, 
        #   validation_data=(x_val, y_val)
        validation_split=0.3, 
          )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=0)    # 1/1을 없앨 수 있다.
results = model.predict([18])

print("로스는 : ", loss)
print("18의 예측값 : ", results)
print("걸린시간 : ", round(end_time - start_time, 2), "초")   # round는 반올림, 2는 소수 2째자리까지 반올림