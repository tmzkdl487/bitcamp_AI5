import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(777)
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

dataset = load_digits()
x_data, y_data = dataset.data, dataset.target

one_hot = OneHotEncoder(sparse=False)
y_data = one_hot.fit_transform(y_data.reshape(-1, 1))

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 64])
w = tf.compat.v1.Variable(tf.random_normal([64, 10], name='weight'))
b = tf.compat.v1.Variable(tf.zeros([1, 10], name='bais'))
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

#2. 모델
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x, w) + b)

#3-1. 컴파일    # Categorical_CrossEntropy
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis = 1))

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 20001
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b],
                                         feed_dict={x:x_data, y:y_data})
    if step % 200 ==0:
        print(step, 'loss: ', cost_val)
        
#4.평가, 예측
y_predict = sess.run(hypothesis, feed_dict={x:x_data})
y_predict = np.argmax(y_predict, 1)
y_data = np.argmax(y_data, 1)

acc = accuracy_score(y_predict, y_data)
print('acc : ', acc)    

sess.close()

# acc :  0.09905397885364496