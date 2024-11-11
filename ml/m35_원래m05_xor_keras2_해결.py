import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([0, 1, 1, 0])
print(x_data.shape, y_data.shape)   # (4, 2) (4,)

#2. 모델 
# model = LinearSVCR()
# model = Perceptron()

# 내가 만든 모델
# model = Sequential()
# model.add(Dense(64, input_dim=2, activation='sigmoid'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))

# 선생님 보여준 모델
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1,  activation='sigmoid'))

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x_data, y_data, batch_size=12, epochs=100)

#4. 평가, 예측
# acc = model.score(x_data, y_data)
# print('model.score : ', acc)

loss = model.evaluate(x_data, y_data)
print('acc : ', loss[1])

y_predict = model.predict(x_data)
y_predict = np.round(y_predict).reshape(-1,).astype(int)

acc2 = accuracy_score(y_data, y_predict)
print('accuracy_score : ', acc2)

print("================================")
print(y_data)
print(y_predict)

# 모델 건드려서 acc 1.0 만들 것
# acc :  1.0
# accuracy_score :  1.0
# ================================
# [0 1 1 0]
# [0 1 1 0]