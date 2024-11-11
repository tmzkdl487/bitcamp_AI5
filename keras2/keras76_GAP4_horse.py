from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
import numpy as np
import time
import tensorflow as tf

tf.random.set_seed(337)
np.random.seed(337)

# 1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255
)

path_train = './_data/image/horse_human/'

xy_train2 = train_datagen.flow_from_directory(
    path_train, 
    target_size=(32, 32),  # VGG16의 입력 크기와 맞춤
    batch_size=1027, 
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

x_train, x_test, y_train, y_test = train_test_split(xy_train2[0][0], xy_train2[0][1], train_size=0.75, 
                                                     shuffle=True, 
                                                     random_state=337)

# 2. 모델
vgg16 = VGG16(include_top=False, input_shape=(32, 32, 3))
vgg16.trainable = True  # False # 동결건조

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start_time = time.time()

model.fit(x_train, y_train, epochs=10, batch_size=1, validation_split=0.3, verbose=1)

end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

print("76_horse_로스는 : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("걸린시간: ", round(end_time - start_time, 2), "초")

# model.add(Flatten())
# 76_horse_로스는 :  0.6978816390037537
# ACC :  0.471
# 걸린시간:  56.34 초

# model.add(GlobalAveragePooling2D())