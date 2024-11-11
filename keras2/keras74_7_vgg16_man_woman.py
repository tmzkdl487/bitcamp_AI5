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
x_train = np.load(np_path + 'keras45_gender_04_x_train.npy')
y_train = np.load(np_path + 'keras45_gender_04_y_train.npy')
x_test = np.load(np_path + 'keras45_gender_04_x_test.npy')
y_test = np.load(np_path + 'keras45_gender_04_y_test.npy')

x_train = np.repeat(x_train, 3, axis=-1)
x_test = np.repeat(x_test, 3, axis=-1)

# 데이터 스케일링
x_train = x_train / 255.0
x_test = x_test / 255.0

# 데이터 분할
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.75, random_state=337)

# 2. VGG16 모델 설정
vgg16 = VGG16(include_top=False, input_shape=(100, 100, 3))
vgg16.trainable = True  # VGG16의 가중치 동결

# 새 모델 정의
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 3. 모델 컴파일 및 훈련 설정
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['acc'])

# 훈련 시작
start_time = time.time()
history = model.fit(x_train, y_train, epochs=5, batch_size=1, validation_data=(x_val, y_val), 
                     verbose=1)
end_time = time.time()

# 4. 평가 및 예측
loss, acc = model.evaluate(x_test, y_test, verbose=1)

# 5. 예측
y_pred = model.predict(x_test)
y_pred = np.round(y_pred).astype(int)

# 결과 출력
print("man_woman_로스는 : ", loss)
print("ACC : ", round(loss, 3))
print("걸린시간: ", round(end_time - start_time, 2), "초")

####### [실습] ########
# 비교할 것
# 1. 이전에 본인이 한 최상의 결과와.
# 2. 가중치를 동결하지 않고 훈련시켰을때, tranable=True, (디폴트)
# 3. 가중치를 동결하고 훈련시켰을때, tranable=False
####### 위에 2, 3번할때는 time 체크 할 것.

# 1. 이전에 본인이 한 최상의 결과와. 

# 2. 가중치를 동결하지 않고 훈련시켰을때, tranable=True, (디폴트)


# 3. 가중치를 동결하고 훈련시켰을때, tranable=False 

# Traceback (most recent call last):                                                                                                n
#   File "c:\ai5\study\keras2\keras74_7_vgg16_men_women.py", line 54, in <module>                                                   :0 
#     history = model.fit(x_train, y_train, epochs=5, batch_size=1, validation_data=(x_val, y_val),
#   File "C:\anaconda3\envs\tf118gpu\lib\site-packages\keras\utils\traceback_utils.py", line 67, in error_handler
#     raise e.with_traceback(filtered_tb) from None                                                                                 :
#   File "C:\anaconda3\envs\tf118gpu\lib\site-packages\tensorflow\python\framework\constant_op.py", line 106, in convert_to_eager_tensor
#     return ops.EagerTensor(value, ctx.device_name, dtype)
# tensorflow.python.framework.errors_impl.InternalError: Failed copying input tensor from /job:localhost/replica:0/task:0/device:CPU:0 to /job:localhost/replica:0/task:0/device:GPU:0 in order to run _EagerConst: Dst tensor is not initialized.
