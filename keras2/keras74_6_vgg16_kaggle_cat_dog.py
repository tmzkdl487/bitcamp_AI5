from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import tensorflow as tf

# 랜덤 시드 설정
tf.random.set_seed(337)
np.random.seed(337)

# 1. 데이터 불러오기
np_path = 'c:/ai5/_data/_save_npy/'  # npy 데이터 경로 설정

# 개별 파일 불러오기
x_train_1 = np.load(np_path + 'keras43_01_x_train.npy')
x_train_2 = np.load(np_path + 'keras49_image_cat_dog_01_x_train.npy')

y_train_1 = np.load(np_path + 'keras43_01_y_train.npy')
y_train_2 = np.load(np_path + 'keras49_image_cat_dog_01_y_train.npy')

x_test = np.load(np_path + 'keras43_01_x_test.npy')
y_test = np.load(np_path + 'keras43_01_y_test.npy')

# 데이터를 합치기
x_train = np.concatenate((x_train_1, x_train_2), axis=0)
y_train = np.concatenate((y_train_1, y_train_2), axis=0)

# 데이터 스케일링
x_train = x_train / 255.0
x_test = x_test / 255.0

# 데이터 분할
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.75, random_state=337)

# 2. VGG16 모델 설정
vgg16 = VGG16(include_top=False, input_shape=(80, 80, 3))
vgg16.trainable = True  # VGG16의 가중치 동결

# 새 모델 정의
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 3. 모델 컴파일 및 훈련 설정
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['acc'])

# 훈련 시작
start_time = time.time()
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val), 
                     verbose=1)
end_time = time.time()

# 4. 평가 및 예측
loss, acc = model.evaluate(x_test, y_test, verbose=1)

# 5. 예측
y_pred = model.predict(x_test)
y_pred = np.round(y_pred).astype(int)

# 결과 출력
print("cat_dog_로스는 : ", loss)
# print("ACC : ", round(loss[1], 3))
print("걸린시간: ", round(end_time - start_time, 2), "초")

####### [실습] ########
# 비교할 것
# 1. 이전에 본인이 한 최상의 결과와.
# 2. 가중치를 동결하지 않고 훈련시켰을때, tranable=True, (디폴트)
# 3. 가중치를 동결하고 훈련시켰을때, tranable=False
####### 위에 2, 3번할때는 time 체크 할 것.

# 1. 이전에 본인이 한 최상의 결과와. 

# 2. 가중치를 동결하지 않고 훈련시켰을때, tranable=True, (디폴트)
# cat_dog_로스는 :  0.6928432583808899
# 걸린시간:  349.69 초

# 3. 가중치를 동결하고 훈련시켰을때, tranable=False 
# cat_dog_로스는 :  0.7321468591690063
# 걸린시간:  138.77 초