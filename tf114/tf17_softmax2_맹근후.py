import tensorflow as tf
import numpy as np
tf.set_random_seed(3434)

#1. 데이터
x_data = [[1, 2, 1, 1], 
          [2, 1, 3, 2], 
          [3, 1, 3, 4], 
          [4, 1, 5, 5],
          [1, 7, 5, 5], 
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]
y_data = [[0, 0, 1],    # 2
          [0, 0, 1],
          [0, 0, 1], 
          [0, 1, 0],    # 1
          [0, 1, 0], 
          [0, 1, 0],
          [1, 0, 0],    # 0
          [1, 0, 0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
w = tf.compat.v1.Variable(tf.random_normal([4, 3], name='weight'))
b = tf.compat.v1.Variable(tf.zeros([1, 3], name='bais'))
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

#2. 모델
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x, w) + b)

#3-1. 컴파일    # Categorical_CrossEntropy
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis = 1))

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 20001
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b],
                                         feed_dict={x:x_data, y:y_data})
    if step % 200 ==0:
        print(step, 'loss: ', cost_val)
        
print(w_val)
# [[-5.1505084   0.35491568  3.0000458 ]
#  [ 0.17925632 -0.16369088  0.8599828 ]
#  [ 1.421063    1.1341188  -1.2763014 ]
#  [ 1.0060028   0.27885923 -1.1719328 ]]

print(b_val)
# [[-3.7147024 -0.9961369  4.7108355]]

#4.평가, 예측
y_predict = sess.run(hypothesis, feed_dict={x:x_data})
print(y_predict)
# [[2.1208502e-06 1.4449427e-03 9.9855298e-01]
#  [2.0444966e-06 1.3146541e-01 8.6853254e-01]
#  [4.4279240e-08 1.6361019e-01 8.3638978e-01]
#  [3.5567611e-09 8.8026005e-01 1.1973996e-01]
#  [3.1533283e-01 6.6951251e-01 1.5154636e-02]
#  [1.4924365e-01 8.5072935e-01 2.7016440e-05]
#  [4.7952336e-01 5.2038765e-01 8.9050198e-05]
#  [7.2873205e-01 2.7123764e-01 3.0280476e-05]]
y_predict = np.argmax(y_predict, 1)

print(y_predict)    # [2 2 2 1 1 1 1 0]

y_data = np.argmax(y_data, 1)

print(y_data)

# [2 2 2 1 1 1 0 0]

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict, y_data)
print('acc : ', acc)    # acc :  0.875

sess.close()