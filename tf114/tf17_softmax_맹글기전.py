import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

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

x = tf.compat.v1.placeholder(tf.float32, shape = [None, 4]) 
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 3]) 
 
w = tf.compat.v1.Variable(tf.random.normal([4,3]), name = 'weight')     
b = tf.compat.v1.Variable(tf.random.normal([1,3]), name = 'bias') 

#2. 모델구성
hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)  

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1)) 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04).minimize(loss)  

sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sees:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, loss_val = sess.run([optimizer, loss], feed_dict = {x:x_data, y:y_data})
        if step % 200 ==0:
            print(step, loss_val)
            
    results = sess.run(hypothesis, feed_dict = {x: x_data})    
    print(results, sess.run(tf.arg_max(results, 1)))  

