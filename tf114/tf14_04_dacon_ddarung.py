import tensorflow as tf
tf.compat.v1.set_random_seed(777)
import pandas as pd
from sklearn.metrics import r2_score

#1. 데이터
path = "C://ai5//_data//dacon//따릉이//"
train_csv = pd.read_csv(path + "train.csv")

train_csv = train_csv.dropna()

x_data = train_csv.drop(['count'], axis =1).values
y_data = train_csv['count'].values.reshape(-1, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_data.shape[1]], name='weight')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random.normal([x_data.shape[1], 1]), name='weight')  
b = tf.compat.v1.Variable(tf.compat.v1.random.normal([1]), name='bias') 

#2. 모델
hypothesis = tf.compat.v1.matmul(x, w) + b

#3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-7)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs=1001
for step in range(epochs):
    cost_val, _ = sess.run([loss, train], feed_dict={x: x_data, y: y_data})
    if step % 100 ==0:
        print(step, 'loss : ', cost_val)

y_pred = sess.run(hypothesis, feed_dict={x: x_data})

test_loss = sess.run(loss, feed_dict={x: x_data, y: y_data})
r2 = r2_score(y_data, y_pred)
print(f"Test Loss: {test_loss}")
print(f"R2 Score: {r2}")

sess.close()