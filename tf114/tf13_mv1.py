import tensorflow as tf
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

tf.compat.v1.set_random_seed(777)

# 1. 데이터
x1_data = [73., 93., 89., 96., 73.]     # 국어
x2_data = [80., 88., 91., 98., 66.]     # 영어
x3_data = [75., 93., 90., 100., 70.]    # 수학
y_data = [152., 185., 180., 196., 142.] # 환산점수

# [실습] 맹그러봐!!!

x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w1 = tf.compat.v1.Variable(tf.random.normal([1], mean=0.0, stddev=0.001), dtype=tf.float32, name='weight1')
w2 = tf.compat.v1.Variable(tf.random.normal([1], mean=0.0, stddev=0.001), dtype=tf.float32, name='weight2')
w3 = tf.compat.v1.Variable(tf.random.normal([1], mean=0.0, stddev=0.001), dtype=tf.float32, name='weight3')
b = tf.compat.v1.Variable(tf.random.normal([1], mean=0.0, stddev=0.001), dtype=tf.float32, name='bias')

#2. 모델
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

#3-1. 컴파일 
loss = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00001)   # <- Adam쓰고 싶으면 GradientDescentOptimizer을 Adam으로 바꾸면 됨.
train = optimizer.minimize(loss)

# 기록을 위한 리스트
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2. 훈련
for step in range(1000):  # 10,000번 반복
    _, loss_val = sess.run([train, loss], feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y: y_data})
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss_val}")
    loss_history.append(loss_val)

#4. 예측
y_pred = sess.run(hypothesis, feed_dict={x1: x1_data, x2: x2_data, x3: x3_data})
print("Predictions:", y_pred)

r2 = r2_score(y_data, y_pred)
mae = mean_absolute_error(y_data, y_pred)
print("R2 Score:", r2)
print("Mean Absolute Error:", mae)

sess.close()

plt.plot(range(len(loss_history)), loss_history)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Loss over Training')
plt.grid()
plt.show()

# R2 Score: 0.9974503579628946
# Mean Absolute Error: 0.9716766357421875