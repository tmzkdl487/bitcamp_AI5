import tensorflow as tf
import pandas as pd
from sklearn.metrics import r2_score

tf.compat.v1.set_random_seed(777)

#1. 데이터
path = 'C://ai5/_data/dacon/diabetes/'
train_csv = pd.read_csv(path + "train.csv")

# print(train_csv.columns)

train_csv = train_csv.drop(columns=['ID'])

x_data = train_csv.drop(columns=['Outcome']).values.astype('float32')
y_data = train_csv['Outcome'].values.reshape(-1, 1).astype('float32')

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 1], name='weight'))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1], name='bais'))

#2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)

#3-1. 컴파일
loss = -tf.reduce_mean(y*tf.math.log(hypothesis)+(1-y)*tf.math.log(1-hypothesis))  # binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(2001):
    cost_val, _ = sess.run([loss, train], feed_dict={x: x_data, y: y_data})
    if step % 20 == 0:
        print(step, 'loss', cost_val)
        
#4. 평가, 예측
# x_test = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
x_test = x_data

y_pred = sess.run(hypothesis, feed_dict={x: x_test})
y_predict = sess.run(tf.cast(y_pred > 0.5, dtype=tf.float32))

print(y_predict)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_data, y_predict)
print('acc : ', acc)   

sess.close()