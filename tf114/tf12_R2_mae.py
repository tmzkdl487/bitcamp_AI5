import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.set_random_seed(777)

#1. 데이터
x_train = [1,2,3,4,5]
y_train = [1,2,3,4,5]
x_test = [6,7,8]
y_test = [6,7,8]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')
b = tf.compat.v1.Variable([0], dtype=tf.float32, name='bias')


#2. 모델
hypothesis = x * w + b
#3-1. 컴파일 ## model.compile(loss='mse', optimizer ='sad')
loss = tf.reduce_mean(tf.square(hypothesis - y))

############### 옵티마이저 ###################
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)   # <- Adam쓰고 싶으면 GradientDescentOptimizer을 Adam으로 바꾸면 됨.
# train = optimizer.minimize(loss)

lr = 0.1
gradient = tf.reduce_mean(( x*w +b - y ) * x)

descent = w - lr * gradient

train = w.assign(descent)

############### 옵티마이저 끝 ###################

w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2. 훈련
for step in range(31):
    _, loss_v, w_v = sess.run([train, loss, w], feed_dict={x: x_train, y:y_train})
    print(step, '\t', loss_v, '/t', w_v)
    
    w_history.append(w_v[0])
    loss_history.append(loss_v)

print("---------------------- W history --------------------------")
print(w_history)
print("---------------------- loss history --------------------------")
print(loss_history)

# plt.plot(loss_history)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.grid()
# plt.show()
    
###### [실습] R2, mae 맹그러!!!!!! #############
# from sklearn.metrics import r2_score, mean_absolute_error

# y_pred = sess.run(hypothesis, feed_dict={x: x_test})
# print("Predictions:", y_pred)

# r2 = r2_score(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# print("R2 Score:", r2)
# print("Mean Absolute Error:", mae)

sess.close()

# 결과
# Predictions: [6. 7. 8.]
# R2 Score: 1.0
# Mean Absolute Error: 0.0

###### [실습] R2, mae 맹그러!!!!!! - 선생님 버전 #############
from sklearn.metrics import r2_score, mean_absolute_error

y_predict = x_test* w_v
print('y_predict : ', y_predict)

r2 = r2_score(y_test, y_predict)
print('r2: ', r2)

mae = mean_absolute_error(y_test, y_predict)
print('mae: ', mae)

# 결과
# y_predict :  [6. 7. 8.]
# r2:  1.0
# mae:  0.0