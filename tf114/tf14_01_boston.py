# <회귀>
# 01_boston.py
# 02_california.py    (8, 1) 인풋, 아웃풋
# 03_diabetes.py      (10, 1)
# 04_dacon_ddarung.py (8, 2)
# 05_kaggle_bike.py   (8, 1)

import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from tensorflow.keras.datasets import boston_housing
from sklearn.metrics import r2_score

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# print(x_train.shape, y_train.shape) # (404, 13) (404,)
# print(x_test.shape, y_test.shape)   # (102, 13) (102,)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# [실습] 맹그러봐
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([13, 1], name='weight'))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1], name='bais'))

#2. 모델
hypothesis = tf.compat.v1.matmul(x, w) + b

#3-1 컴파일
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis-y))    #mse

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-6) 
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs=1001
for step in range(epochs):
    cost_val, _ = sess.run([loss, train], feed_dict={x: x_train, y: y_train})
    if step % 100 ==0:
        print(step, 'loss : ', cost_val)

y_pred = sess.run(hypothesis, feed_dict={x: x_test})

test_loss = sess.run(loss, feed_dict={x: x_test, y: y_test})
r2 = r2_score(y_test, y_pred)
print(f"Test Loss: {test_loss}")
print(f"R2 Score: {r2}")

sess.close()

# Test Loss: 157.64898681640625
# R2 Score: -0.8938215645868142