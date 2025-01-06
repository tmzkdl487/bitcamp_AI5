import tensorflow as tf
tf.compat.v1.set_random_seed(1100)

#1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]   # (4, 2)
y_data = [[0], [1], [1], [0]]

#2. 모델
# layer1 :  model.add(Dence(10, input_dim=2))
x = tf.compat.v1.placeholder(tf.float32, shape= [None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape= [None, 1])

w1 = tf.compat.v1.Variable(tf.random.normal([2, 10], name='weight1'))
b1 = tf.compat.v1.Variable(tf.zeros([10], name='bais1'))

# hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)

layer1 = tf.compat.v1.matmul(x, w1) + b1    # (N, 10)

# layer2 : model.add(Dense(5, input_dim=10))
w2 = tf.compat.v1.Variable(tf.random.normal([10, 5], name='weight2'))
b2 = tf.compat.v1.Variable(tf.zeros([5], name='bais2'))
layer2 = tf.compat.v1.matmul(layer1, w2) +b2    # (5, 3)

# layer3 : model.add(Dense(3, input_dim=5))
w3 = tf.compat.v1.Variable(tf.random.normal([5, 3], name='weight3'))
b3 = tf.compat.v1.Variable(tf.zeros([3], name='bais3'))
layer3 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer2, w3) +b3)   # (N, 3)

# layer4 : model.add(Dense(4, input_dim=3))
w4 = tf.compat.v1.Variable(tf.random.normal([3, 2], name='weight4'))
b4 = tf.compat.v1.Variable(tf.zeros([2], name='bais4'))
layer4 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer3, w4) +b4)   # (N, 4)

# output : model.add(Dense(1, activation='sigmoid'))
w5 = tf.compat.v1.Variable(tf.random.normal([2, 1], name='weight5'))
b5 = tf.compat.v1.Variable(tf.zeros([1], name='bais5'))
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer4, w5) + b5)

#3-1. 컴파일
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))  # binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(101):
    cost_val, _ = sess.run([loss, train], feed_dict={x: x_data, y: y_data})
    if step % 10 == 0:
        print(step, 'loss', cost_val)
        
#4. 평가, 예측
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])

y_pred = sess.run(hypothesis, feed_dict={x: x_data})
y_predict = sess.run(tf.cast(y_pred > 0.5, dtype=tf.float32), feed_dict={x_test:x_data})

print(y_predict)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_data, y_predict)
print('acc : ', acc) 

# acc :  0.75